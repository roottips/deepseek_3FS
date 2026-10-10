/**
 * =========================================================================
 * File:  src/common/net/ib/GpuMemMgr.cc
 * Purpose: Implementation of GPU device-memory allocation + RDMA pinning
 *          helpers for 3FS's GDR adaptation, using the CUDA VMM API
 *          (cuMemAddressReserve / cuMemCreate / cuMemMap / cuMemSetAccess).
 *
 * Notes:
 *   * All CUDA driver API calls are guarded by probe() so that 3FS built
 *     without CUDA still links (the symbols are weak-loaded via dlopen).
 *   * Verbs registration calls reuse 3FS's existing IBDevice::regMemory()
 *     for the PEERMEM path -- that path treats GPU pointers transparently
 *     once nvidia-peermem is loaded; only the DMABUF path needs the
 *     ibv_reg_dmabuf_mr entry point.
 *   * VMM allocations (cuMemCreate) are the only allocation kind that
 *     reliably supports cuMemGetHandleForAddressRange (dma-buf fd export).
 *     This is why we do NOT use cuMemAlloc/cudaMalloc.
 * =========================================================================
 */

#include "GpuMemMgr.h"

#include "IBDevice.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>

#include <cstring>
#include <stdexcept>

namespace hf3fs::net::ib {

namespace {
// Weakly-loaded CUDA driver entry points; we never want to take a hard
// link dependency on libcuda when GPUDirect is unused.
struct CudaApi {
  void *handle{nullptr};

  // Core driver
  CUresult (*cuInit)(int){nullptr};
  CUresult (*cuDriverGetVersion)(int *){nullptr};
  CUresult (*cuDeviceGet)(CUdevice *, int){nullptr};

  // Pointer attributes
  CUresult (*cuPointerGetAttribute)(void *, CUpointer_attribute,
                                     CUdeviceptr){nullptr};
  CUresult (*cuPointerSetAttribute)(const void *, CUpointer_attribute,
                                     CUdeviceptr){nullptr};

  // VMM: address management
  CUresult (*cuMemAddressReserve)(CUdeviceptr *, size_t, size_t,
                                   CUdeviceptr, CUmemAllocationFlags){nullptr};
  CUresult (*cuMemAddressFree)(CUdeviceptr, size_t){nullptr};

  // VMM: physical allocation
  CUresult (*cuMemCreate)(CUmemGenericAllocationHandle *, size_t,
                           const CUmemAllocationProp *,
                           const CUmemAllocationFlags *){nullptr};
  CUresult (*cuMemRelease)(CUmemGenericAllocationHandle){nullptr};

  // VMM: mapping
  CUresult (*cuMemMap)(CUdeviceptr, size_t, size_t,
                        CUmemGenericAllocationHandle, CUmemFlags){nullptr};
  CUresult (*cuMemUnmap)(CUdeviceptr, size_t){nullptr};
  CUresult (*cuMemSetAccess)(CUdeviceptr, size_t,
                              const CUmemAccessDesc *, size_t){nullptr};

  // VMM: IPC handle export/import
  CUresult (*cuMemExportToShareableHandle)(void *,
                                             CUmemGenericAllocationHandle,
                                             CUmemAllocationHandleType,
                                             CUmemFlags){nullptr};
  CUresult (*cuMemImportFromShareableHandle)(CUmemGenericAllocationHandle *,
                                                void *,
                                                CUmemAllocationHandleType){nullptr};

  // dma-buf fd export for RDMA registration
  CUresult (*cuMemGetHandleForAddressRange)(void *, CUdeviceptr, size_t,
                                              CUmemRangeHandleType,
                                              CUmemFlags){nullptr};

  // Error string helper
  CUresult (*cuGetErrorString)(CUresult, const char **){nullptr};

  // Legacy cuMemAlloc/cuMemFree kept as fallback when VMM is unavailable.
  CUresult (*cuMemAlloc)(CUdeviceptr *, size_t){nullptr};
  CUresult (*cuMemFree)(CUdeviceptr){nullptr};

  bool hasVmm() const {
    return cuMemAddressReserve && cuMemCreate && cuMemMap &&
           cuMemSetAccess && cuMemAddressFree && cuMemRelease;
  }

  static const CudaApi &get();
};

const CudaApi &CudaApi::get() {
  static CudaApi api;
  static std::once_flag once;
  std::call_once(once, []() {
    api.handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!api.handle) {
      return;
    }
#define SYM(name, type, sym) \
    api.name = (type)dlsym(api.handle, sym)

    SYM(cuInit, CUresult(*)(int), "cuInit");
    SYM(cuDriverGetVersion, CUresult(*)(int *), "cuDriverGetVersion");
    SYM(cuDeviceGet, CUresult(*)(CUdevice *, int), "cuDeviceGet");
    SYM(cuPointerGetAttribute,
        CUresult(*)(void *, CUpointer_attribute, CUdeviceptr),
        "cuPointerGetAttribute");
    SYM(cuPointerSetAttribute,
        CUresult(*)(const void *, CUpointer_attribute, CUdeviceptr),
        "cuPointerSetAttribute");
    SYM(cuMemAddressReserve,
        CUresult(*)(CUdeviceptr *, size_t, size_t, CUdeviceptr,
                     CUmemAllocationFlags),
        "cuMemAddressReserve");
    SYM(cuMemAddressFree,
        CUresult(*)(CUdeviceptr, size_t),
        "cuMemAddressFree");
    SYM(cuMemCreate,
        CUresult(*)(CUmemGenericAllocationHandle *, size_t,
                     const CUmemAllocationProp *,
                     const CUmemAllocationFlags *),
        "cuMemCreate");
    SYM(cuMemRelease,
        CUresult(*)(CUmemGenericAllocationHandle),
        "cuMemRelease");
    SYM(cuMemMap,
        CUresult(*)(CUdeviceptr, size_t, size_t,
                     CUmemGenericAllocationHandle, CUmemFlags),
        "cuMemMap");
    SYM(cuMemUnmap,
        CUresult(*)(CUdeviceptr, size_t),
        "cuMemUnmap");
    SYM(cuMemSetAccess,
        CUresult(*)(CUdeviceptr, size_t,
                     const CUmemAccessDesc *, size_t),
        "cuMemSetAccess");
    SYM(cuMemExportToShareableHandle,
        CUresult(*)(void *, CUmemGenericAllocationHandle,
                     CUmemAllocationHandleType, CUmemFlags),
        "cuMemExportToShareableHandle");
    SYM(cuMemImportFromShareableHandle,
        CUresult(*)(CUmemGenericAllocationHandle *, void *,
                     CUmemAllocationHandleType),
        "cuMemImportFromShareableHandle");
    SYM(cuMemGetHandleForAddressRange,
        CUresult(*)(void *, CUdeviceptr, size_t,
                     CUmemRangeHandleType, CUmemFlags),
        "cuMemGetHandleForAddressRange");
    SYM(cuGetErrorString,
        CUresult(*)(CUresult, const char **),
        "cuGetErrorString");
    SYM(cuMemAlloc, CUresult(*)(CUdeviceptr *, size_t), "cuMemAlloc");
    SYM(cuMemFree, CUresult(*)(CUdeviceptr), "cuMemFree");
#undef SYM
    if (api.cuInit) api.cuInit(0);
  });
  return api;
}

// Helper: convert a CUresult to a human-readable string for logging.
const char *cudaErrorString(CUresult rc) {
  const char *s = nullptr;
  const auto &api = CudaApi::get();
  if (api.cuGetErrorString) {
    api.cuGetErrorString(rc, &s);
  }
  return s ? s : "unknown CUDA error";
}

}  // namespace

// ---- GpuVmmAllocation lifecycle ----------------------------------------

int GpuVmmAllocation::exportIpcFd() {
  if (ipc_fd >= 0) return ipc_fd;
  const auto &api = CudaApi::get();
  if (!api.cuMemExportToShareableHandle || !alloc_handle) {
    return -1;
  }
  // CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1
  int fd = -1;
  CUresult rc = api.cuMemExportToShareableHandle(
      &fd, alloc_handle,
      static_cast<CUmemAllocationHandleType>(1),  /*flags=*/0);
  if (rc != CUDA_SUCCESS) {
    return -1;
  }
  ipc_fd = fd;
  return ipc_fd;
}

std::shared_ptr<GpuVmmAllocation> GpuVmmAllocation::importFromIpcFd(
    int ipc_fd, size_t size, int target_device_id) {
  const auto &api = CudaApi::get();
  if (!api.hasVmm() || !api.cuMemImportFromShareableHandle) {
    return nullptr;
  }

  // Round up to 2 MiB granularity.
  size = (size + kGpuVmmGranularity - 1) & ~(kGpuVmmGranularity - 1);

  // 1. Reserve a VA range in this process.
  CUdeviceptr va = 0;
  CUresult rc = api.cuMemAddressReserve(
      &va, size, /*alignment=*/0, /*fixedAddress=*/0,
      /*flags=*/0);
  if (rc != CUDA_SUCCESS) return nullptr;

  // 2. Import the allocation handle from the POSIX fd.
  CUmemGenericAllocationHandle alloc_handle = 0;
  rc = api.cuMemImportFromShareableHandle(
      &alloc_handle, reinterpret_cast<void *>(static_cast<intptr_t>(ipc_fd)),
      static_cast<CUmemAllocationHandleType>(1));
  if (rc != CUDA_SUCCESS) {
    api.cuMemAddressFree(va, size);
    return nullptr;
  }

  // 3. Map the physical allocation into our reserved VA.
  rc = api.cuMemMap(va, size, /*offset=*/0, alloc_handle, /*flags=*/0);
  if (rc != CUDA_SUCCESS) {
    api.cuMemRelease(alloc_handle);
    api.cuMemAddressFree(va, size);
    return nullptr;
  }

  // 4. Grant the target GPU read/write access.
  CUmemAccessDesc desc = {};
  desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  desc.location.id = static_cast<CUdevice>(target_device_id);
  desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  rc = api.cuMemSetAccess(va, size, &desc, 1);
  if (rc != CUDA_SUCCESS) {
    api.cuMemUnmap(va, size);
    api.cuMemRelease(alloc_handle);
    api.cuMemAddressFree(va, size);
    return nullptr;
  }

  auto alloc = std::make_shared<GpuVmmAllocation>();
  alloc->va = va;
  alloc->size = size;
  alloc->alloc_handle = alloc_handle;
  alloc->device_id = target_device_id;
  alloc->ipc_fd = -1;  // imported, not owned
  return alloc;
}

void GpuVmmAllocation::destroy() {
  const auto &api = CudaApi::get();
  if (!api.hasVmm()) return;
  if (va) {
    if (api.cuMemUnmap) api.cuMemUnmap(va, size);
    if (alloc_handle) api.cuMemRelease(alloc_handle);
    api.cuMemAddressFree(va, size);
    va = 0;
    alloc_handle = 0;
  }
  if (ipc_fd >= 0) {
    close(ipc_fd);
    ipc_fd = -1;
  }
}

// ---- GpuMemMgr static helpers ------------------------------------------

bool GpuMemMgr::probe() {
  const auto &api = CudaApi::get();
  if (!api.handle || !api.cuDriverGetVersion) return false;
  int version = 0;
  api.cuDriverGetVersion(&version);
  return version > 0;
}

bool GpuMemMgr::isDevicePointer(void *ptr) {
  const auto &api = CudaApi::get();
  if (!api.cuPointerGetAttribute) return false;
  CUmemorytype mt = static_cast<CUmemorytype>(0);
  // CU_POINTER_ATTRIBUTE_MEMORY_TYPE == 0
  CUresult rc = api.cuPointerGetAttribute(
      &mt, static_cast<CUpointer_attribute>(0),
      reinterpret_cast<CUdeviceptr>(ptr));
  if (rc != CUDA_SUCCESS) return false;
  return mt == CU_MEMORYTYPE_DEVICE;
}

uint64_t GpuMemMgr::bufferId(void *ptr) {
  const auto &api = CudaApi::get();
  if (!api.cuPointerGetAttribute) return 0;
  uint64_t bid = 0;
  // CU_POINTER_ATTRIBUTE_BUFFER_ID == 4
  CUresult rc = api.cuPointerGetAttribute(
      &bid, static_cast<CUpointer_attribute>(4),
      reinterpret_cast<CUdeviceptr>(ptr));
  if (rc != CUDA_SUCCESS) return 0;
  return bid;
}

bool GpuMemMgr::setSyncMemops(void *ptr) {
  const auto &api = CudaApi::get();
  if (!api.cuPointerSetAttribute) return false;
  unsigned int flag = 1;
  // CU_POINTER_ATTRIBUTE_SYNC_MEMOPS == 0x40000015 (annex)
  CUresult rc = api.cuPointerSetAttribute(
      &flag, static_cast<CUpointer_attribute>(0x40000015),
      reinterpret_cast<CUdeviceptr>(ptr));
  return rc == CUDA_SUCCESS;
}

int GpuMemMgr::exportDmabufFd(void *gpu_ptr, size_t nbytes) {
  const auto &api = CudaApi::get();
  if (!api.cuMemGetHandleForAddressRange) return -1;
  int fd = -1;
  // CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD == 0x01
  CUresult rc = api.cuMemGetHandleForAddressRange(
      &fd, reinterpret_cast<CUdeviceptr>(gpu_ptr), nbytes,
      static_cast<CUmemRangeHandleType>(0x01), 0);
  if (rc != CUDA_SUCCESS) return -1;
  return fd;
}

// ---- VMM allocation API ------------------------------------------------

std::shared_ptr<GpuVmmAllocation> GpuMemMgr::vmmAlloc(
    size_t size, int device_id, bool export_ipc) {
  const auto &api = CudaApi::get();
  if (!api.hasVmm()) {
    // Fallback: use legacy cuMemAlloc and wrap it in a GpuVmmAllocation
    // with alloc_handle=0 so the caller knows to skip dmabuf export.
    if (!api.cuMemAlloc) return nullptr;
    size = (size + kGpuVmmGranularity - 1) & ~(kGpuVmmGranularity - 1);
    CUdeviceptr ptr = 0;
    if (api.cuMemAlloc(&ptr, size) != CUDA_SUCCESS) return nullptr;
    auto alloc = std::make_shared<GpuVmmAllocation>();
    alloc->va = ptr;
    alloc->size = size;
    alloc->alloc_handle = 0;  // legacy path; no handle to release
    alloc->device_id = device_id;
    setSyncMemops(reinterpret_cast<void *>(ptr));
    return alloc;
  }

  // Round up to 2 MiB granularity.
  size = (size + kGpuVmmGranularity - 1) & ~(kGpuVmmGranularity - 1);

  // 1. Reserve a virtual address range (no physical pages yet).
  CUdeviceptr va = 0;
  CUresult rc = api.cuMemAddressReserve(
      &va, size, /*alignment=*/0, /*fixedAddress=*/0, /*flags=*/0);
  if (rc != CUDA_SUCCESS) {
    return nullptr;
  }

  // 2. Create a physical allocation handle on the target GPU.
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = static_cast<CUdevice>(device_id);
  // Request READWRITE access so the NIC (via peermem) and local GPU
  // can both access the pages.
  CUmemAllocationFlags flags = {};
  CUmemGenericAllocationHandle alloc_handle = 0;
  rc = api.cuMemCreate(&alloc_handle, size, &prop, &flags);
  if (rc != CUDA_SUCCESS) {
    api.cuMemAddressFree(va, size);
    return nullptr;
  }

  // 3. Map the physical allocation into the reserved VA.
  rc = api.cuMemMap(va, size, /*offset=*/0, alloc_handle, /*flags=*/0);
  if (rc != CUDA_SUCCESS) {
    api.cuMemRelease(alloc_handle);
    api.cuMemAddressFree(va, size);
    return nullptr;
  }

  // 4. Grant the owning GPU read/write access.
  CUmemAccessDesc access_desc = {};
  access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access_desc.location.id = static_cast<CUdevice>(device_id);
  access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  rc = api.cuMemSetAccess(va, size, &access_desc, 1);
  if (rc != CUDA_SUCCESS) {
    api.cuMemUnmap(va, size);
    api.cuMemRelease(alloc_handle);
    api.cuMemAddressFree(va, size);
    return nullptr;
  }

  // 5. SYNC_MEMOPS must be set before any verbs call touches this VA.
  setSyncMemops(reinterpret_cast<void *>(va));

  // 6. [optional] Export IPC fd for cross-process sharing.
  int ipc_fd = -1;
  if (export_ipc) {
    // CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1
    ipc_fd = -1;
    rc = api.cuMemExportToShareableHandle(
        &ipc_fd, alloc_handle,
        static_cast<CUmemAllocationHandleType>(1), /*flags=*/0);
    if (rc != CUDA_SUCCESS) {
      ipc_fd = -1;
      // Non-fatal: continue without IPC; the buffer is still usable
      // in-process.
    }
  }

  auto alloc = std::make_shared<GpuVmmAllocation>();
  alloc->va = va;
  alloc->size = size;
  alloc->alloc_handle = alloc_handle;
  alloc->device_id = device_id;
  alloc->ipc_fd = ipc_fd;
  return alloc;
}

std::shared_ptr<GpuVmmAllocation> GpuMemMgr::vmmImportFromIpcFd(
    int ipc_fd, size_t size, int target_device_id) {
  return GpuVmmAllocation::importFromIpcFd(ipc_fd, size, target_device_id);
}

// ---- GpuMemMgr instance API --------------------------------------------

GpuMemMgr &GpuMemMgr::instance() {
  static GpuMemMgr mgr;
  return mgr;
}

GpuMemMgr::GpuMemMgr() = default;

GpuMrCache &GpuMemMgr::cacheFor(IBDevice *dev) {
  std::lock_guard<std::mutex> g(lock_);
  auto it = caches_.find(dev);
  if (it == caches_.end()) {
    auto cache = std::make_unique<GpuMrCache>(dev);
    auto [inserted, _] = caches_.emplace(dev, std::move(cache));
    return *inserted->second;
  }
  return *it->second;
}

GpuBufferPool *GpuMemMgr::poolFor(size_t slot_bytes) {
  std::lock_guard<std::mutex> g(lock_);
  auto it = pools_.find(slot_bytes);
  return it == pools_.end() ? nullptr : it->second.get();
}

GpuBufferPool *GpuMemMgr::ensurePool(size_t slot_bytes, size_t num_slots,
                                      int device_id) {
  std::lock_guard<std::mutex> g(lock_);
  auto &slot = pools_[slot_bytes];
  if (!slot) {
    slot = GpuBufferPool::create(slot_bytes, num_slots, device_id);
  }
  return slot.get();
}

// ---- GpuBufferPool -----------------------------------------------------

std::unique_ptr<GpuBufferPool> GpuBufferPool::create(size_t slot_bytes,
                                                       size_t num_slots,
                                                       int device_id) {
  size_t total = slot_bytes * num_slots;
  // Use VMM allocation (export IPC so the user app can import).
  auto vmm = GpuMemMgr::vmmAlloc(total, device_id, /*export_ipc=*/true);
  if (!vmm || !vmm->valid()) {
    return nullptr;
  }
  auto pool = std::unique_ptr<GpuBufferPool>(
      new GpuBufferPool(slot_bytes, num_slots, device_id, vmm));
  // Populate the free list with slot-aligned addresses.
  for (size_t i = 0; i < num_slots; ++i) {
    pool->free_list_.push_back(
        reinterpret_cast<void *>(vmm->va + i * slot_bytes));
  }
  return pool;
}

GpuBufferPool::~GpuBufferPool() {
  if (vmm_) {
    vmm_->destroy();
    vmm_.reset();
  }
}

void *GpuBufferPool::acquire() {
  std::lock_guard<std::mutex> g(free_list_lock_);
  if (free_list_.empty()) return nullptr;
  void *slot = free_list_.back();
  free_list_.pop_back();
  return slot;
}

void GpuBufferPool::release(void *slot) {
  std::lock_guard<std::mutex> g(free_list_lock_);
  free_list_.push_back(slot);
}

// ---- GpuMrCache --------------------------------------------------------

GpuMrCache::GpuMrCache(IBDevice *dev, size_t max_pinned_bytes)
    : dev_(dev), max_pinned_bytes_(max_pinned_bytes) {}

GpuMrCache::~GpuMrCache() { teardown(); }

std::shared_ptr<GpuMrHandle> GpuMrCache::registerRegion(void *gpu_ptr,
                                                        size_t nbytes) {
  // Sanity: must actually be device memory; silently fall back to host
  // path if not (RDMABuf handles that case directly).
  if (!GpuMemMgr::isDevicePointer(gpu_ptr)) {
    return nullptr;
  }
  // SYNC_MEMOPS is mandatory per NVIDIA docs.
  GpuMemMgr::setSyncMemops(gpu_ptr);
  uint64_t bid = GpuMemMgr::bufferId(gpu_ptr);

  uint64_t key = reinterpret_cast<uintptr_t>(gpu_ptr) ^ (nbytes << 33) ^
                 (bid << 17);
  std::lock_guard<std::mutex> g(lock_);
  auto it = entries_.find(key);
  if (it != entries_.end()) {
    it->second.first->refcount++;
    lru_order_.splice(lru_order_.end(), lru_order_, it->second.second);
    return it->second.first;
  }
  evictUntilFits_(nbytes);

  // Try dma-buf first when available; it avoids the kernel-module
  // dependency on CUDA 13+. Requires a VMM-backed allocation.
  auto handle = registerDmabufPath_(gpu_ptr, nbytes);
  if (!handle) {
    handle = registerPeerMemPath_(gpu_ptr, nbytes);
  }
  if (!handle) {
    return nullptr;
  }
  handle->buffer_id = bid;
  handle->refcount = 1;
  lru_order_.emplace_back(reinterpret_cast<uintptr_t>(gpu_ptr), nbytes, bid);
  auto list_it = std::prev(lru_order_.end());
  entries_.emplace(key, std::make_pair(handle, list_it));
  pinned_bytes_.fetch_add(nbytes, std::memory_order_relaxed);
  return handle;
}

std::shared_ptr<GpuMrHandle> GpuMrCache::registerPeerMemPath_(
    void *gpu_ptr, size_t nbytes) {
  // Reuses IBDevice::regMemory(); when nvidia-peermem is loaded this
  // transparently pins GPU pages into BAR1.
  ibv_mr *mr = dev_->regMemory(gpu_ptr, nbytes, kGpuMrAccessFlags);
  if (!mr) {
    return nullptr;
  }
  auto h = std::make_shared<GpuMrHandle>();
  h->kind = GpuMrHandle::Kind::PEERMEM;
  h->addr = gpu_ptr;
  h->nbytes = nbytes;
  h->lkey = mr->lkey;
  h->rkey = mr->rkey;
  h->mr = mr;
  h->dmabuf_fd = -1;
  h->buffer_id = 0;
  h->refcount = 0;
  return h;
}

std::shared_ptr<GpuMrHandle> GpuMrCache::registerDmabufPath_(
    void *gpu_ptr, size_t nbytes) {
  int fd = GpuMemMgr::exportDmabufFd(gpu_ptr, nbytes);
  if (fd < 0) {
    return nullptr;
  }
  // Use the dmabuf-aware verbs entry point. ibv_reg_dmabuf_mr is part of
  // libibverbs >= 1.8; we dlsym it so older runtimes fall back silently.
  static auto p_reg_dmabuf_mr =
      (ibv_mr * (*)(ibv_pd *, uint64_t, size_t, uint64_t, int, int))
          dlsym(RTLD_DEFAULT, "ibv_reg_dmabuf_mr");
  if (!p_reg_dmabuf_mr) {
    close(fd);
    return nullptr;
  }
  ibv_pd *pd = dev_->pd();
  ibv_mr *mr = p_reg_dmabuf_mr(
      pd,
      /*offset=*/0, /*length=*/nbytes,
      /*iova=*/reinterpret_cast<uint64_t>(gpu_ptr),
      fd, kGpuMrAccessFlags);
  if (!mr) {
    close(fd);
    return nullptr;
  }
  auto h = std::make_shared<GpuMrHandle>();
  h->kind = GpuMrHandle::Kind::DMABUF;
  // Advertise the device VA, NOT mr->addr (which is meaningless for
  // dma-buf MRs). Peers must target remote_addr = device VA.
  h->addr = gpu_ptr;
  h->nbytes = nbytes;
  h->lkey = mr->lkey;
  h->rkey = mr->rkey;
  h->mr = mr;
  h->dmabuf_fd = fd;
  h->buffer_id = 0;
  h->refcount = 0;
  return h;
}

void GpuMrCache::release(const std::shared_ptr<GpuMrHandle> &handle) {
  if (!handle) return;
  std::lock_guard<std::mutex> g(lock_);
  if (handle->refcount > 0) {
    handle->refcount--;
  }
  // Lazy: do not deregister; eviction will reclaim when needed.
}

void GpuMrCache::invalidateStale() {
  std::lock_guard<std::mutex> g(lock_);
  for (auto it = entries_.begin(); it != entries_.end();) {
    auto &handle = it->second.first;
    uint64_t cur_bid = GpuMemMgr::bufferId(handle->addr);
    if (cur_bid != handle->buffer_id) {
      deregister_(handle.get());
      pinned_bytes_.fetch_sub(handle->nbytes, std::memory_order_relaxed);
      lru_order_.erase(it->second.second);
      it = entries_.erase(it);
    } else {
      ++it;
    }
  }
}

void GpuMrCache::evictUntilFits_(size_t needed) {
  while (pinned_bytes_.load(std::memory_order_relaxed) + needed >
             max_pinned_bytes_ &&
         !lru_order_.empty()) {
    auto front_it = lru_order_.begin();
    auto [ptr, nbytes, bid] = *front_it;
    uint64_t key = ptr ^ (nbytes << 33) ^ (bid << 17);
    auto entry_it = entries_.find(key);
    if (entry_it == entries_.end()) {
      lru_order_.erase(front_it);
      continue;
    }
    auto &handle = entry_it->second.first;
    if (handle->refcount > 0) {
      // Don't evict in-use regions; stop trying.
      break;
    }
    deregister_(handle.get());
    pinned_bytes_.fetch_sub(nbytes, std::memory_order_relaxed);
    lru_order_.erase(front_it);
    entries_.erase(entry_it);
  }
}

void GpuMrCache::deregister_(GpuMrHandle *h) {
  if (h->mr) {
    ibv_dereg_mr(h->mr);
    h->mr = nullptr;
  }
  if (h->dmabuf_fd >= 0) {
    close(h->dmabuf_fd);
    h->dmabuf_fd = -1;
  }
}

void GpuMrCache::teardown() {
  std::lock_guard<std::mutex> g(lock_);
  for (auto &[key, val] : entries_) {
    deregister_(val.first.get());
  }
  entries_.clear();
  lru_order_.clear();
  pinned_bytes_.store(0, std::memory_order_relaxed);
}

}  // namespace hf3fs::net::ib
