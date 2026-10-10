#include "Shm.h"

#include <fcntl.h>
#include <cstdlib>
#include <folly/experimental/coro/BlockingWait.h>
#include <numa.h>
#include <sys/mman.h>
#include <unistd.h>

#include "common/net/ib/GpuMemMgr.h"

namespace hf3fs::lib {

// GDR config flags (env-overridable; see hf3fs/docs/USRBIO.md).
static bool s_use_gpu_memory = [] {
  const char *e = std::getenv("HF3FS_IOV_USE_GPU_MEMORY");
  return e != nullptr && std::string(e) == "1";
}();
static size_t s_gpu_device_id = [] {
  const char *e = std::getenv("HF3FS_IOV_GPU_DEVICE_ID");
  return e ? std::stoul(e) : 0;
}();

ShmBuf::ShmBuf(const Path &p, size_t sz, size_t bsz, int numa, meta::Uid u, int pr, int pp)
    : path(p),
      size(sz),
      blockSize(bsz ? bsz : sz),
      off(0),
      user(u),
      pid(pr),
      ppid(pp),
      id(Uuid::random()),
      owner_(true),
      numaNode_(numa),
      memhs_((size + blockSize - 1) / blockSize) {
  if (bsz < 0 || bsz > sz) {
    throw std::invalid_argument(fmt::format("invalid block size {} total bytes {}", bsz, sz));
  }

  mapBuf();

  if (numaNode_ >= 0) {
    numa_tonode_memory(bufStart, size, numaNode_);
  }
}

ShmBuf::ShmBuf(const Path &p, off_t o, size_t sz, size_t bsz, Uuid u)
    : path(p),
      size(sz),
      blockSize(bsz ? bsz : sz),
      off(o),
      //      user(0),
      id(u),
      owner_(false),
      numaNode_(-1),
      memhs_((size + blockSize - 1) / blockSize) {
  if (bsz < 0 || bsz > sz) {
    throw std::invalid_argument(fmt::format("invalid block size {} total bytes {}", bsz, sz));
  }

  XLOGF(DBG, "this {} buf start {} off {}", (void *)this, (void *)bufStart, off);

  mapBuf();

  XLOGF(DBG, "this {} buf start {} off {}", (void *)this, (void *)bufStart, off);
}

ShmBuf::ShmBuf(const ghbm_desc_t &d, int dmabuf_fd)
    : size(d.size),
      blockSize(d.block_size),
      gpuBacked(true),
      dmabufFd(dmabuf_fd),
      ghbmDesc(d) {
  // 注意：这里**故意不设置 bufStart**，除非 ATS。
  // 普通 PCIe GPU 上，gpu_va 属于客户端进程地址空间，daemon 解引用 = SIGSEGV。
  bufStart = d.ats ? reinterpret_cast<uint8_t *>(d.ats_cpu_va) : nullptr;
  id = Uuid::fromBytes(d.uuid);   // 让 checkId() / IovTable 的 key 语义保持一致
}

ShmBuf::~ShmBuf() {
  XLOGF(DBG, "calling dtor of shm {}", (void *)this);

  folly::coro::blockingWait(deregisterForIO());

  // if (owner_) {
  //   auto res = ftruncate(fd_, 0);
  //   XLOGF_IF(ERR, res < 0, "ftruncate shm file to 0 failed {}", res);
  // }

  unmapBuf();
}

CoTask<void> ShmBuf::deregisterForIO() {
  XLOGF(DBG, "regging {} memhs {}", regging_.load(), memhs_.size());

  if (!regging_) {
    co_return;
  }

  if (!memhs_[0].load()) {
    co_await memhBaton_;
  }

  regging_ = false;

  for (auto &memh : memhs_) {
    memh.store(std::shared_ptr<storage::client::IOBuffer>());
  }

  co_return;
}

CoTask<void> ShmBuf::registerForIO(folly::Executor::KeepAlive<> exec,
                                   storage::client::StorageClient &sc,
                                   std::function<void()> &&recordMetrics) {
  if (regging_) {
    co_await memhBaton_;
  }

  regging_ = true;
  memhBaton_.reset();

  for (auto &memh : memhs_) {
    memh.store(std::shared_ptr<storage::client::IOBuffer>());
  }

  auto f = [this, &sc, recordMetrics = std::move(recordMetrics)]() {
    for (size_t i = 0; i < memhs_.size(); ++i) {
      auto res = sc.registerIOBuffer(bufStart + blockSize * i, std::min(size - blockSize * i, blockSize));
      if (!res) {
        XLOGF(ERR,
              "failed to register buffer @{} seg #{} with bytes {} block size {} code {} msg {}",
              (void *)bufStart,
              i,
              size,
              blockSize,
              res.error().code(),
              res.error().message());
      } else {
        memhs_[i].store(std::make_shared<storage::client::IOBuffer>(std::move(*res)));
        recordMetrics();
      }
    }

    memhBaton_.post();
  };
  if (exec) {
    exec.get()->add(std::move(f));
  } else {
    f();
  }
}

void ShmBuf::unmapBuf() {
  if (is_gpu_backed_) {
    return unmapBufGpu_();
  }
  if (bufStart) {
    munmap(bufStart, size);

    if (owner_) {
      shm_unlink(path.c_str());
    }

    bufStart = nullptr;
  }
}

CoTask<std::shared_ptr<storage::client::IOBuffer>> ShmBuf::memh(size_t off) {
  auto idx = off / blockSize;
  if (regging_ && !memhs_[idx].load()) {
    co_await memhBaton_;
  }

  co_return memhs_[idx].load();
}

void ShmBuf::mapBuf() {
  if (s_use_gpu_memory) {
    return mapBufGpu_();
  }

  auto fd = shm_open(path.c_str(), O_RDWR | (owner_ ? O_CREAT | O_EXCL : 0), 0666);
  if (fd < 0) {
    auto err = errno;
    throw std::runtime_error(fmt::format("failed to shm_open {} is owner {} errno {} euid {} uid {}",
                                         path.native(),
                                         owner_,
                                         err,
                                         geteuid(),
                                         getuid()));
  }

  SCOPE_EXIT { close(fd); };

  if (owner_) {
    auto res = ftruncate(fd, size);
    if (res < 0) {
      throw std::runtime_error("failed to ftruncate");
    }
  }
  bufStart = (uint8_t *)mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, off);
  if (bufStart == MAP_FAILED) {
    throw std::runtime_error("failed to mmap");
  }

  XLOGF(DBG, "this {} owner {} buf start {} size {} off {}", (void *)this, owner_, (void *)bufStart, size, off);
}

// ---- NEW: GPU-backed mapBuf (CUDA VMM API) ---------------------------
//
// Allocates the whole iov ring as a single VMM region on the target GPU:
//   cuMemAddressReserve -> cuMemCreate -> cuMemMap -> cuMemSetAccess
// via hf3fs::net::GpuMemMgr::vmmAlloc(). The returned GpuVmmAllocation
// is RAII: dropping the shared_ptr reverses the mapping.

void ShmBuf::mapBufGpu_() {
  using namespace hf3fs::net;

  if (!GpuMemMgr::probe()) {
    throw std::runtime_error(
        "HF3FS_IOV_USE_GPU_MEMORY=1 but CUDA is unavailable");
  }

  // VMM allocation with IPC fd export so a different process (the user
  // application) can import the same physical allocation.
  vmm_alloc_ = GpuMemMgr::vmmAlloc(
      size, static_cast<int>(s_gpu_device_id), /*export_ipc=*/true);
  if (!vmm_alloc_ || !vmm_alloc_->valid()) {
    throw std::runtime_error(
        "GpuMemMgr::vmmAlloc failed for USRBIO iov ring "
        "(cuMemAddressReserve/cuMemCreate/cuMemMap)");
  }

  // Point bufStart at the VMM-mapped device VA so the rest of ShmBuf's
  // existing I/O path (registerForIO, memh, ShmBufForIO::ptr) works
  // unchanged.
  bufStart = reinterpret_cast<uint8_t *>(vmm_alloc_->va);
  is_gpu_backed_ = true;

  XLOGF(DBG, "ShmBuf GPU-backed (VMM), bufStart {} size {} dev {}",
        (void *)bufStart, size, s_gpu_device_id);
}

void ShmBuf::unmapBufGpu_() {
  if (vmm_alloc_) {
    // RAII: dropping the shared_ptr runs GpuVmmAllocation::destroy()
    // which reverses the VMM mapping in order:
    //   cuMemUnmap -> cuMemRelease -> cuMemAddressFree
    vmm_alloc_.reset();
    bufStart = nullptr;
    is_gpu_backed_ = false;
  }
}

}  // namespace hf3fs::lib
