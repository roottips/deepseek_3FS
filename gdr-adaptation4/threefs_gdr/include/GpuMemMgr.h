/**
 * =========================================================================
 * File:  src/common/net/ib/GpuMemMgr.h
 * Purpose: GPU device-memory allocator + RDMA-pinning helper for 3FS.
 *
 * Baseline: 3FS commit f84e609692dded708f9c910caf5c5af3444e1024
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
 *      a) VMM allocations are the only path that reliably supports
 *         cuMemGetHandleForAddressRange (dma-buf fd export for RDMA).
 *      b) VMM gives explicit control over address alignment (2 MiB
 *         granularity), which matches the GPU BAR1 page granularity.
 *      c) VMM allocations can be exported as POSIX file-descriptor IPC
 *         handles (cuMemExportToShareableHandle) for cross-process
 *         sharing.
 * 2. cuPointerSetAttribute(SYNC_MEMOPS=1) for every allocation.
 * 3. Two registration paths, auto-selected per allocation:
 *    a) PEERMEM path: ibv_reg_mr() on the device VA.
 *    b) DMABUF path:  cuMemGetHandleForAddressRange() + ibv_reg_dmabuf_mr().
 * 4. LRU pin cache to amortize the millisecond-scale BAR1 page-pinning
 *    cost across many small transfers reusing the same buffer.
 *
 * Integration with real 3FS (f84e609):
 *   * Namespace is hf3fs::net (matches IBDevice/RDMABuf).
 *   * IBDevice::pd(), name(), regMemory(), deregMemory() are already
 *     public in the baseline; we only add bar1Budget().
 *   * IBDevice::all() returns const std::vector<IBDevice::Ptr>&.
 *   * RDMABuf::Inner::mrs_ is an array<ibv_mr*, kMaxDeviceCnt> (one MR
 *     per device), NOT a single mr_. registerMemory() takes no args and
 *     iterates IBDevice::all() internally.
 * =========================================================================
 */

#pragma once

#include <cuda.h>
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

namespace hf3fs::net {

class IBDevice;

// Mirrors NVIDIA's recommended access flag set for bi-directional GPU RDMA.
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
  CUdeviceptr va{0};
  size_t size{0};
  CUmemGenericAllocationHandle alloc_handle{0};
  int device_id{-1};
  int ipc_fd{-1};

  bool valid() const { return va != 0; }

  int exportIpcFd();
  static std::shared_ptr<GpuVmmAllocation> importFromIpcFd(
      int ipc_fd, size_t size, int target_device_id);
  void destroy();
};

/**
 * Single pinned GPU memory region. Registered against one IBDevice.
 */
struct GpuMrHandle {
  enum class Kind : uint8_t { PEERMEM = 0, DMABUF = 1 };
  Kind kind;
  void *addr;          // device VA (VMM-mapped)
  size_t nbytes;
  uint32_t lkey;
  uint32_t rkey;
  ibv_mr *mr;          // underlying verbs MR (nullptr after deregister)
  int dmabuf_fd;       // -1 unless DMABUF path; kept open for MR lifetime
  uint64_t buffer_id;  // CUDA BUFFER_ID; used to detect re-allocation
  uint32_t refcount;

  bool isValid() const { return mr != nullptr; }
};

/**
 * Pool of fixed-size GPU buffers backed by a single VMM allocation.
 */
class GpuBufferPool {
 public:
  static std::unique_ptr<GpuBufferPool> create(size_t slot_bytes,
                                              size_t num_slots,
                                              int device_id);
  ~GpuBufferPool();

  void *acquire();
  void release(void *slot);
  size_t slotBytes() const { return slot_bytes_; }
  int deviceId() const { return device_id_; }
  void *base() const { return reinterpret_cast<void *>(vmm_->va); }
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
 * LRU pin cache: (ptr, nbytes, buffer_id) -> GpuMrHandle, per IBDevice.
 */
class GpuMrCache {
 public:
  explicit GpuMrCache(IBDevice *dev,
                      size_t max_pinned_bytes = 8ULL * (1ULL << 30));
  ~GpuMrCache();

  std::shared_ptr<GpuMrHandle> registerRegion(void *gpu_ptr, size_t nbytes);
  void release(const std::shared_ptr<GpuMrHandle> &handle);
  void invalidateStale();
  size_t pinnedBytes() const {
    return pinned_bytes_.load(std::memory_order_relaxed);
  }
  void teardown();

 private:
  IBDevice *dev_;
  size_t max_pinned_bytes_;
  std::atomic<size_t> pinned_bytes_{0};

  mutable std::mutex lock_;
  std::list<std::tuple<uintptr_t, size_t, uint64_t>> lru_order_;
  std::unordered_map<uint64_t,
                      std::pair<std::shared_ptr<GpuMrHandle>,
                                std::list<std::tuple<uintptr_t, size_t, uint64_t>>::iterator>>
      entries_;

  std::shared_ptr<GpuMrHandle> registerPeerMemPath_(void *gpu_ptr, size_t nbytes);
  std::shared_ptr<GpuMrHandle> registerDmabufPath_(void *gpu_ptr, size_t nbytes);
  void evictUntilFits_(size_t needed);
  static void deregister_(GpuMrHandle *h);
};

/**
 * Top-level GPU memory subsystem.
 */
class GpuMemMgr {
 public:
  static GpuMemMgr &instance();

  static bool probe();
  static bool isDevicePointer(void *ptr);
  static uint64_t bufferId(void *ptr);
  static bool setSyncMemops(void *ptr);
  static int exportDmabufFd(void *gpu_ptr, size_t nbytes);

  static std::shared_ptr<GpuVmmAllocation> vmmAlloc(
      size_t size, int device_id, bool export_ipc = false);
  static std::shared_ptr<GpuVmmAllocation> vmmImportFromIpcFd(
      int ipc_fd, size_t size, int target_device_id);

  GpuMrCache &cacheFor(IBDevice *dev);
  GpuBufferPool *poolFor(size_t slot_bytes);
  GpuBufferPool *ensurePool(size_t slot_bytes, size_t num_slots,
                            int device_id);

 private:
  GpuMemMgr();
  std::mutex lock_;
  std::unordered_map<IBDevice *, std::unique_ptr<GpuMrCache>> caches_;
  std::unordered_map<size_t, std::unique_ptr<GpuBufferPool>> pools_;
};

}  // namespace hf3fs::net
