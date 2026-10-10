/**
 * =========================================================================
 * File:  src/common/net/ib/GpuMemMgr.h
 * Purpose: GPU device-memory allocator + RDMA-pinning helper for 3FS.
 *
 * Background
 * ----------
 * 3FS today allocates its client I/O buffers via POSIX shared memory
 * (shm_open + mmap). Storage servers push data over RDMA into that
 * host-resident memory, then the user application must cudaMemcpy() it
 * to the GPU before any CUDA kernel can consume it.
 *
 * With GPUDirect RDMA (GDR), the storage server can RDMA-WRITE directly
 * into GPU device memory, eliminating the host-side hop. This file
 * provides the GPU memory allocation + verbs registration plumbing so
 * 3FS's existing RDMABuf/IBDevice/StorageClient stack can be reused
 * unchanged when the underlying buffer happens to live in VRAM.
 *
 * Design choices (VMM-based allocation)
 * -------------------------------------
 * 1. CUDA Virtual Memory Management (VMM) API for all allocations:
 *    cuMemAddressReserve -> cuMemCreate -> cuMemMap -> cuMemSetAccess.
 *    This replaces cudaMalloc/cuMemAlloc because:
 *      a) VMM allocations are backed by the CUDA VMM, which is the only
 *         path that supports cuMemGetHandleForAddressRange (dma-buf fd
 *         export for RDMA registration). cuMemAlloc-returned pointers
 *         may or may not be VMM-backed depending on the runtime.
 *      b) VMM gives explicit control over address alignment (2 MiB
 *         granularity), which matches the GPU BAR1 page granularity and
 *         maximizes BAR mapping efficiency.
 *      c) VMM allocations can be exported as POSIX file-descriptor IPC
 *         handles (cuMemExportToShareableHandle) so the FUSE process
 *         can share the iov buffer with a different user application
 *         process via cuMemImportFromShareableHandle + cuMemMap.
 * 2. cuPointerSetAttribute(SYNC_MEMOPS=1) for every allocation -- without
 *    this the NIC may observe a stale read before the producing CUDA
 *    kernel flushes to BAR1.
 * 3. Two registration paths, auto-selected per allocation:
 *    a) PEERMEM path: ibv_reg_mr() on the device VA. Requires
 *       `modprobe nvidia-peermem` (R470+ driver).
 *    b) DMABUF path:  cuMemGetHandleForAddressRange() + ibv_reg_dmabuf_mr().
 *       CUDA 13+, no kernel module needed. Requires VMM-backed allocation.
 * 4. LRU pin cache to amortize the (millisecond-scale) BAR1 page-pinning
 *    cost across many small transfers reusing the same buffer.
 * 5. Allocation pool for fixed-size GPU buffers to avoid per-I/O
 *    allocation churn.
 *
 * Forward declares the IBDevice friend used for the registration call.
 * =========================================================================
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <infiniband/verbs.h>

#include <atomic>
#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace hf3fs::net::ib {

class IBDevice;

// Mirrors NVIDIA's recommended access flag set for bi-directional GPU RDMA.
// RELAXED_ORDERING is critical for write-combining BAR1 throughput.
constexpr int kGpuMrAccessFlags =
    IBV_ACCESS_LOCAL_WRITE |
    IBV_ACCESS_REMOTE_WRITE |
    IBV_ACCESS_REMOTE_READ |
    IBV_ACCESS_RELAXED_ORDERING;

// BAR1 page granularity on NVIDIA GPUs.
constexpr size_t kGpuBar1PageSize = 1ULL << 16;  // 64 KiB

// VMM allocation granularity (2 MiB). cuMemCreate requires size to be
// a multiple of this value.
constexpr size_t kGpuVmmGranularity = 1ULL << 21;

/**
 * RAII wrapper around a single VMM allocation.
 *
 * Lifecycle:
 *   1. cuMemAddressReserve()  -> reserve a VA range (no physical pages)
 *   2. cuMemCreate()           -> create a physical allocation handle
 *   3. cuMemMap()              -> map the physical handle into the VA range
 *   4. cuMemSetAccess()        -> grant the target GPU read/write access
 *   5. [optional] cuMemExportToShareableHandle() -> IPC fd for cross-process
 *   6. [optional] cuMemGetHandleForAddressRange() -> dma-buf fd for RDMA
 *
 * On destruction the steps are reversed:
 *   cuMemUnmap -> cuMemRelease -> cuMemAddressFree
 */
struct GpuVmmAllocation {
  CUdeviceptr va{0};                          // reserved virtual address
  size_t size{0};                             // size in bytes (VMM-aligned)
  CUmemGenericAllocationHandle alloc_handle{0}; // physical allocation handle
  int device_id{-1};                          // owning GPU
  int ipc_fd{-1};                             // POSIX fd for cross-process share
  // dmabuf fd is not stored here; it is created on demand by GpuMrCache
  // and kept open for the lifetime of the registered MR.

  bool valid() const { return va != 0; }

  // Export a POSIX file-descriptor IPC handle so another process can
  // import this allocation via cuMemImportFromShareableHandle + cuMemMap.
  // Returns -1 on failure or if already exported (idempotent: returns
  // the cached fd).
  int exportIpcFd();

  // Import an allocation from a POSIX fd exported by another process.
  // Performs steps 1-4 (reserve VA, import handle, map, set access) on
  // the current process. Returns a valid GpuVmmAllocation on success.
  static std::shared_ptr<GpuVmmAllocation> importFromIpcFd(
      int ipc_fd, size_t size, int target_device_id);

  // Tear down in reverse order. Safe to call once; idempotent.
  void destroy();
};

/**
 * Single pinned GPU memory region.
 *
 * `addr` is the device virtual address. For the PEERMEM path this equals
 * mr->addr; for the DMABUF path mr->addr is the *ibverbs* internal
 * address (==0 or iova) and must NOT be advertised to peers; advertise
 * `addr` (=device VA) instead. Callers should always use `addr` to
 * build remote_addr fields.
 */
struct GpuMrHandle {
  enum class Kind : uint8_t { PEERMEM = 0, DMABUF = 1 };
  Kind kind;
  void *addr;          // device VA (VMM-mapped)
  size_t nbytes;
  uint32_t lkey;       // local access key
  uint32_t rkey;       // remote access key (advertise to peers)
  ibv_mr *mr;          // underlying verbs MR (nullptr after deregister)
  int dmabuf_fd;       // -1 unless DMABUF path; kept open for MR lifetime
  uint64_t buffer_id;  // CUDA BUFFER_ID; used to detect re-allocation
  uint32_t refcount;

  bool isValid() const { return mr != nullptr; }
};

/**
 * Pool of fixed-size GPU buffers backed by a single VMM allocation.
 *
 * One large VMM allocation is subdivided into ``num_slots`` slots of
 * ``slot_bytes`` each. Pinning happens at the big-allocation granularity,
 * not per slot, so the per-I/O cost is O(1).
 */
class GpuBufferPool {
 public:
  static std::unique_ptr<GpuBufferPool> create(size_t slot_bytes,
                                              size_t num_slots,
                                              int device_id);

  ~GpuBufferPool();

  // Acquire one slot; returns the device VA. Returns nullptr if pool empty.
  void *acquire();
  void release(void *slot);
  size_t slotBytes() const { return slot_bytes_; }
  int deviceId() const { return device_id_; }
  void *base() const { return reinterpret_cast<void *>(vmm_->va); }

  // Expose the IPC fd so a different process can map the same allocation.
  int exportIpcFd() const { return vmm_->ipc_fd; }

 private:
  GpuBufferPool(size_t slot_bytes, size_t num_slots, int device_id,
                std::shared_ptr<GpuVmmAllocation> vmm)
      : slot_bytes_(slot_bytes),
        num_slots_(num_slots),
        device_id_(device_id),
        vmm_(std::move(vmm)) {}

  size_t slot_bytes_;
  size_t num_slots_;
  int device_id_;
  std::shared_ptr<GpuVmmAllocation> vmm_;
  std::mutex free_list_lock_;
  std::vector<void *> free_list_;
};

/**
 * LRU pin cache: (ptr, nbytes, buffer_id) -> GpuMrHandle.
 *
 * Re-registering GPU pages is millisecond-scale work. 3FS's USRBIO ring
 * reuses the same iov buffers for thousands of I/Os, so the cache makes
 * per-I/O cost ~0 after the first miss.
 *
 * Eviction policy: when the configured BAR1 budget is exceeded, the
 * least-recently-used entries whose refcount==0 are deregistered.
 */
class GpuMrCache {
 public:
  explicit GpuMrCache(IBDevice *dev,
                      size_t max_pinned_bytes = 8ULL * (1ULL << 30));  // 8 GiB
  ~GpuMrCache();

  // Pin one GPU buffer (or sub-region thereof). Thread-safe; refcount-aware.
  // Returns nullptr if pinning failed (peermem missing, BAR1 exhausted).
  std::shared_ptr<GpuMrHandle> registerRegion(void *gpu_ptr, size_t nbytes);

  // Release a refcount on a previously registered region. Lazy: MR is kept
  // alive until LRU eviction actually needs the BAR space back.
  void release(const std::shared_ptr<GpuMrHandle> &handle);

  // Drop any cached entry whose underlying CUDA BUFFER_ID has changed
  // (buffer freed + re-allocated at the same VA). Called periodically
  // from the storage client.
  void invalidateStale();

  size_t pinnedBytes() const {
    return pinned_bytes_.load(std::memory_order_relaxed);
  }

  // Manually tear down (process shutdown or role switch).
  void teardown();

 private:
  IBDevice *dev_;
  size_t max_pinned_bytes_;
  std::atomic<size_t> pinned_bytes_{0};

  mutable std::mutex lock_;
  // ordered: most-recently-used at the tail.
  std::list<std::tuple<uintptr_t, size_t, uint64_t>> lru_order_;
  std::unordered_map<uint64_t,
                      std::pair<std::shared_ptr<GpuMrHandle>,
                                std::list<std::tuple<uintptr_t, size_t, uint64_t>>::iterator>>
      entries_;  // key = (ptr ^ (nbytes << 33)) for collisions; see impl

  std::shared_ptr<GpuMrHandle> registerPeerMemPath_(void *gpu_ptr, size_t nbytes);
  std::shared_ptr<GpuMrHandle> registerDmabufPath_(void *gpu_ptr, size_t nbytes);
  void evictUntilFits_(size_t needed);
  static void deregister_(GpuMrHandle *h);
};

/**
 * Top-level GPU memory subsystem for one IB device.
 *
 * Owns:
 *   * The GpuMrCache for that device.
 *   * Optionally, a pool of fixed-size VMM-backed buffers for the USRBIO
 *     iov ring (so the per-I/O allocation churn is avoided).
 *   * Helpers to set SYNC_MEMOPS, query BUFFER_ID, detect whether a
 *     pointer is device vs host memory, and export dma-buf fds.
 *   * A VMM allocation helper (reserve+create+map+setaccess) used by
 *     ShmBuf::mapBufGpu_ and GpuBufferPool::create.
 *
 * The IBDevice owns one of these lazily (so the verbs path stays
 * unchanged when 3FS is built without CUDA headers).
 */
class GpuMemMgr {
 public:
  static GpuMemMgr &instance();

  // Initialize CUDA driver API + verify GPUDirect RDMA preconditions.
  // Returns false if CUDA unavailable or no GPU; 3FS silently falls
  // back to host-memory I/O in that case.
  static bool probe();

  // CUDA driver attribute helpers (header-inline; trivial wrappers).
  static bool isDevicePointer(void *ptr);
  static uint64_t bufferId(void *ptr);
  static bool setSyncMemops(void *ptr);
  static int exportDmabufFd(void *gpu_ptr, size_t nbytes);

  // ---- VMM allocation API ----

  // Allocate a contiguous GPU device-memory region using the VMM API:
  //   cuMemAddressReserve -> cuMemCreate -> cuMemMap -> cuMemSetAccess.
  //
  // size is rounded up to kGpuVmmGranularity (2 MiB). The returned
  // GpuVmmAllocation is RAII; destruction reverses the mapping.
  //
  // device_id selects the GPU that owns the physical pages. Set
  // export_ipc=true to also export a POSIX fd IPC handle so a different
  // process can import the allocation.
  static std::shared_ptr<GpuVmmAllocation> vmmAlloc(
      size_t size, int device_id, bool export_ipc = false);

  // Import a VMM allocation from a POSIX fd IPC handle exported by
  // another process (e.g. the FUSE daemon exporting to the user app):
  //   cuMemAddressReserve -> cuMemImportFromShareableHandle ->
  //   cuMemMap -> cuMemSetAccess.
  static std::shared_ptr<GpuVmmAllocation> vmmImportFromIpcFd(
      int ipc_fd, size_t size, int target_device_id);

  GpuMrCache &cacheFor(IBDevice *dev);
  GpuBufferPool *poolFor(size_t slot_bytes);

  // Create a fixed-size GPU buffer pool, e.g. for a USRBIO iov ring
  // sized to fit one I/O batch.
  GpuBufferPool *ensurePool(size_t slot_bytes, size_t num_slots,
                            int device_id);

 private:
  GpuMemMgr();
  std::mutex lock_;
  std::unordered_map<IBDevice *, std::unique_ptr<GpuMrCache>> caches_;
  std::unordered_map<size_t, std::unique_ptr<GpuBufferPool>> pools_;
};

}  // namespace hf3fs::net::ib
