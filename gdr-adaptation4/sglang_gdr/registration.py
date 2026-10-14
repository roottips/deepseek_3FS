"""GPU-memory RDMA registration cache.

Pin GPU device memory once, reuse across transfers, evict on BAR1 pressure.

Two registration paths are supported and auto-selected per allocation:

* ``PEERMEM`` - classic ``ibv_reg_mr`` on a ``cudaMalloc``-returned device
  pointer. Requires ``modprobe nvidia-peermem`` (legacy, R470+).
* ``DMABUF``  - ``ibv_reg_dmabuf_mr`` on a dma-buf fd exported via
  ``cuMemGetHandleForAddressRange``. CUDA 13.0+ recommended path, no kernel
  module needed. Requires ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``
  so the CUDA VMM backend can export dma-buf handles.

The cache key is ``(data_ptr, nbytes)`` plus the CUDA ``BUFFER_ID`` so that
buffers freed and re-allocated at the same address are not served stale MRs.
"""

from __future__ import annotations

import ctypes
import enum
import logging
import os
import threading
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class GpuMemoryKind(enum.Enum):
    """Auto-selected per allocation; exposes both the verbs MR handle and the
    device virtual address that must be advertised to peers."""

    PEERMEM = "peermem"
    DMABUF = "dmabuf"

    @property
    def needs_kernel_module(self) -> bool:
        return self is GpuMemoryKind.PEERMEM


@dataclass
class GpuMrHandle:
    """Snapshot of one registered GPU memory region.

    ``addr`` is the *device virtual address* that the peer must target with
    ``remote_addr`` in its RDMA-WRITE WR (NOT ``mr.addr`` which is meaningless
    for dma-buf MRs). ``lkey`` / ``rkey`` are the verbs access keys.
    """

    kind: GpuMemoryKind
    addr: int            # device VA to advertise to peers
    nbytes: int
    lkey: int            # local access key (post_send sge.lkey)
    rkey: int            # remote access key (advertised to peer)
    mr: Any              # underlying ibverbs MR object (opaque)
    buffer_id: int       # CUDA BUFFER_ID, used to detect reuse
    fd: int = -1         # dma-buf fd, kept open for the MR lifetime
    # bookkeeping
    refcount: int = 0

    def sge(self, length: Optional[int] = None, offset: int = 0) -> "SGE":
        """Build an ibv_sge for the verbs layer.

        ``length`` defaults to the full region; ``offset`` is added to
        ``addr`` so callers can issue sub-region transfers.
        """
        return SGE(addr=self.addr + offset,
                   length=self.nbytes if length is None else length,
                   lkey=self.lkey)


@dataclass
class SGE:
    addr: int
    length: int
    lkey: int


# ---- CUDA driver API surface (lazy-loaded) -------------------------------

_cudart = None
_cudrv = None
_lib_ibv = None


def _load_cuda():
    global _cudart, _cudrv
    if _cudart is None:
        _cudart = ctypes.CDLL("libcudart.so")
    if _cudrv is None:
        _cudrv = ctypes.CDLL("libcuda.so.1")
    return _cudart, _cudrv


def _ibv():
    """Load the ibverbs shared object for raw calls when the Python
    ``ibverbs`` package is unavailable. Prefer the high-level wrapper."""
    global _lib_ibv
    if _lib_ibv is None:
        _lib_ibv = ctypes.CDLL("libibverbs.so.1")
    return _lib_ibv


def _set_sync_memops(dev_ptr: int) -> None:
    """Force CUDA-side serialization for device pointers used by RDMA.

    Without this, a NIC completion may be observed before the producing
    CUDA kernel has flushed to BAR, causing the consumer to read stale
    data. Mandatory per NVIDIA GPUDirect-RDMA docs.
    """
    _, cudrv = _load_cuda()
    flag = ctypes.c_uint(1)
    # CU_POINTER_ATTRIBUTE_SYNC_MEMOPS = 0x40000015 (annex)
    attr = ctypes.c_uint(0x40000015)
    ret = cudrv.cuPointerSetAttribute(
        ctypes.byref(flag), attr, ctypes.c_uint64(dev_ptr))
    if ret != 0:
        logger.warning("cuPointerSetAttribute(SYNC_MEMOPS) failed rc=%d", ret)


def _buffer_id(dev_ptr: int) -> int:
    """Return CUDA BUFFER_ID; lets the cache detect re-allocation at the
    same VA and invalidate the stale MR."""
    _, cudrv = _load_cuda()
    bid = ctypes.c_uint64(0)
    # CU_POINTER_ATTRIBUTE_BUFFER_ID = 4
    attr = ctypes.c_uint(4)
    ret = cudrv.cuPointerGetAttribute(
        ctypes.byref(bid), attr, ctypes.c_uint64(dev_ptr))
    if ret != 0:
        # CPU pointers (or unified memory) return non-zero; use 0 sentinel
        return 0
    return int(bid.value)


def _memory_type(dev_ptr: int) -> int:
    """CU_MEMORYTYPE_DEVICE = 2; CPU = 1; host = 4."""
    _, cudrv = _load_cuda()
    mt = ctypes.c_uint(0)
    # CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 0
    attr = ctypes.c_uint(0)
    ret = cudrv.cuPointerGetAttribute(
        ctypes.byref(mt), attr, ctypes.c_uint64(dev_ptr))
    if ret != 0:
        return 1
    return int(mt.value)


def _export_dmabuf_fd(dev_ptr: int, nbytes: int) -> int:
    """Export the GPU allocation as a dma-buf file descriptor.

    Requires CUDA 13.0+ and VMM-backed allocations (set
    ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` before torch import).
    Returns -1 if the platform does not support dma-buf export.
    """
    _, cudrv = _load_cuda()
    fd = ctypes.c_int(-1)
    # CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD = 0x01
    handle_type = ctypes.c_uint(0x01)
    ret = cudrv.cuMemGetHandleForAddressRange(
        ctypes.byref(fd),
        ctypes.c_uint64(dev_ptr),
        ctypes.c_size_t(nbytes),
        handle_type,
        ctypes.c_uint(0))  # flags, reserved
    if ret != 0:
        logger.debug("cuMemGetHandleForAddressRange failed rc=%d", ret)
        return -1
    return int(fd.value)


class GpuMrCache:
    """LRU registration cache keyed by ``(ptr, nbytes, buffer_id)``.

    Why a cache: pinning GPU pages to BAR1 is millisecond-scale work.
    SGLang reuses the same KV-cache slots across requests (page pool), so
    each (ptr, nbytes) pair is hit thousands of times. Re-registering would
    dominate transfer latency; the cache amortizes to ~0 after first miss.

    Eviction: when BAR1 capacity is exhausted (heuristic: tracked bytes),
    oldest entries are deregistered. Callers must drop stale handles via
    :meth:`invalidate` when the underlying tensor is freed.
    """

    # 64KB BAR1 page alignment, per NVIDIA BAR mapping granularity.
    _ALIGN = 1 << 16

    def __init__(self, pd: Any,
                 max_pinned_bytes: int = 8 * (1 << 30),  # 8 GiB BAR1 budget
                 prefer_dmabuf: Optional[bool] = None):
        self._pd = pd
        self._max = max_pinned_bytes
        self._pinned = 0
        self._prefer_dmabuf = (
            prefer_dmabuf
            if prefer_dmabuf is not None
            else (int(os.environ.get("SGLANG_GDR_PREFER_DMABUF", "0")) == 1)
        )
        # ordered: most-recently-used at the tail
        self._entries: "OrderedDict[Tuple[int, int, int], GpuMrHandle]" = OrderedDict()
        self._lock = threading.RLock()
        # weakref finalizers registered for tensors we have pinned, so we
        # can drop the MR when PyTorch reclaims the buffer.
        self._finalizers: List[weakref.finalize] = []

    # ---- public API ------------------------------------------------------

    def register(self, tensor: Any,
                 access_flags: Optional[Any] = None) -> GpuMrHandle:
        """Register a CUDA tensor (or raw (ptr, nbytes) pair).

        Returns a :class:`GpuMrHandle` that is safe to cache and reuse until
        the tensor is freed or the cache evicts it. Thread-safe.
        """
        ptr, nbytes = self._resolve(tensor)
        bid = _buffer_id(ptr)
        key = (ptr, nbytes, bid)
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                entry.refcount += 1
                self._entries.move_to_end(key)
                return entry
            entry = self._do_register(ptr, nbytes, bid, access_flags)
            self._maybe_evict_locked(nbytes)
            self._entries[key] = entry
            entry.refcount = 1
            self._pinned += nbytes
            return entry

    def register_many(self, tensors: List[Any]) -> List[GpuMrHandle]:
        """Batch convenience; opens one critical section."""
        return [self.register(t) for t in tensors]

    def release(self, handle: GpuMrHandle) -> None:
        """Decrement refcount; do not eagerly deregister (lazy LRU keeps MR)."""
        with self._lock:
            key = (handle.addr, handle.nbytes, handle.buffer_id)
            entry = self._entries.get(key)
            if entry is None:
                return
            entry.refcount = max(0, entry.refcount - 1)

    def invalidate(self, tensor: Any) -> None:
        """Drop a stale entry when the underlying buffer is freed/reallocated."""
        ptr, nbytes = self._resolve(tensor)
        bid = _buffer_id(ptr)
        key = (ptr, nbytes, bid)
        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is not None:
                self._pinned -= entry.nbytes
                self._deregister_locked(entry)

    def invalidate_stale(self) -> None:
        """Scan all cached entries; deregister any whose CUDA BUFFER_ID
        has changed (buffer freed + re-allocated at the same VA)."""
        with self._lock:
            stale_keys = []
            for key, handle in self._entries.items():
                ptr = key[0]
                cur_bid = _buffer_id(ptr)
                if cur_bid != handle.buffer_id:
                    stale_keys.append(key)
            for key in stale_keys:
                entry = self._entries.pop(key)
                self._pinned -= entry.nbytes
                self._deregister_locked(entry)

    def pinned_bytes(self) -> int:
        with self._lock:
            return self._pinned

    def teardown(self) -> None:
        with self._lock:
            for entry in list(self._entries.values()):
                self._deregister_locked(entry)
            self._entries.clear()
            self._pinned = 0
            for f in self._finalizers:
                try:
                    f()
                except Exception:
                    pass
            self._finalizers.clear()

    # ---- internals -------------------------------------------------------

    @staticmethod
    def _resolve(tensor_or_pair: Any) -> Tuple[int, int]:
        if isinstance(tensor_or_pair, tuple):
            return int(tensor_or_pair[0]), int(tensor_or_pair[1])
        # PyTorch / cupy / jax-compatible
        ptr = int(tensor_or_pair.data_ptr())
        nbytes = int(tensor_or_pair.numel() * tensor_or_pair.element_size())
        return ptr, nbytes

    def _do_register(self, ptr: int, nbytes: int, bid: int,
                     access_flags: Optional[Any]) -> GpuMrHandle:
        # Sanity: this must be device memory; otherwise we'd silently pin
        # host RAM through the same path and lose GDR benefits.
        if _memory_type(ptr) != 2:
            raise ValueError(
                f"ptr 0x{ptr:x} is not CUDA device memory (mem_type="
                f"{_memory_type(ptr)}); GDR requires device allocations")

        # Force CUDA/NIC ordering before any verbs call so kernels producing
        # the KV-cache are visible to a NIC about to DMA them out.
        _set_sync_memops(ptr)

        flags = access_flags if access_flags is not None else _default_access()

        # Try dma-buf first (preferred on CUDA 13+).
        if self._prefer_dmabuf:
            fd = _export_dmabuf_fd(ptr, nbytes)
            if fd >= 0:
                return self._register_dmabuf_locked(ptr, nbytes, fd, flags, bid)

        # Fall back to the nvidia-peermem path (works on any CUDA >= 11.4 as
        # long as ``modprobe nvidia-peermem`` is loaded by the operator).
        return self._register_peermem_locked(ptr, nbytes, flags, bid)

    def _register_peermem_locked(self, ptr: int, nbytes: int,
                                 flags: Any, bid: int) -> GpuMrHandle:
        try:
            mr = self._pd.reg_mr(ptr, nbytes, flags)
        except Exception as exc:
            raise RuntimeError(
                f"ibv_reg_mr on GPU ptr 0x{ptr:x} failed. "
                f"Is `nvidia-peermem` loaded? (modprobe nvidia-peermem). "
                f"Underlying error: {exc}") from exc
        return GpuMrHandle(
            kind=GpuMemoryKind.PEERMEM,
            addr=ptr,            # peermem path: mr.addr == dev VA
            nbytes=nbytes,
            lkey=mr.lkey,
            rkey=mr.rkey,
            mr=mr, buffer_id=bid)

    def _register_dmabuf_locked(self, ptr: int, nbytes: int, fd: int,
                                 flags: Any, bid: int) -> GpuMrHandle:
        try:
            # offset=0 within the fd, iova = device VA so peers target it
            mr = self._pd.reg_dmabuf_mr(
                offset=0, length=nbytes, iova=ptr, fd=fd, access=flags)
        except Exception as exc:
            os.close(fd)
            raise RuntimeError(
                f"ibv_reg_dmabuf_mr failed for dev ptr 0x{ptr:x}: {exc}") from exc
        return GpuMrHandle(
            kind=GpuMemoryKind.DMABUF,
            addr=ptr,            # advertise device VA, NOT mr.addr (==0)
            nbytes=nbytes,
            lkey=mr.lkey,
            rkey=mr.rkey,
            mr=mr, buffer_id=bid, fd=fd)

    def _maybe_evict_locked(self, needed: int) -> None:
        """Evict LRU entries until the BAR1 budget can fit ``needed`` bytes."""
        while self._pinned + needed > self._max and self._entries:
            _, victim = self._entries.popitem(last=False)
            if victim.refcount > 0:
                # Don't evict in-use regions; put back and stop.
                self._entries[(victim.addr, victim.nbytes, victim.buffer_id)] = victim
                break
            self._pinned -= victim.nbytes
            self._deregister_locked(victim)

    @staticmethod
    def _deregister_locked(handle: GpuMrHandle) -> None:
        try:
            handle.mr.dereg()
        except Exception:
            pass
        if handle.fd >= 0:
            try:
                os.close(handle.fd)
            except OSError:
                pass
            handle.fd = -1


def _default_access():
    """Access flags required for bi-directional GPU RDMA.

    LOCAL_WRITE  - needed to issue RDMA-WRITE on this MR
    REMOTE_WRITE - lets peers RDMA-WRITE into us (decode side)
    REMOTE_READ  - lets peers RDMA-READ from us (pull mode, optional)
    RELAXED_ORDERING - enables PCIe relaxed ordering, important for
                       write-combining BAR1 throughput.
    """
    try:
        import ibverbs as ib
        return (ib.AccessFlags.LOCAL_WRITE
                | ib.AccessFlags.REMOTE_WRITE
                | ib.AccessFlags.REMOTE_READ
                | ib.AccessFlags.RELAXED_ORDERING)
    except ImportError:
        # Raw libibverbs numeric values when the Python wrapper is absent.
        IBV_ACCESS_LOCAL_WRITE = 1
        IBV_ACCESS_REMOTE_WRITE = 1 << 1
        IBV_ACCESS_REMOTE_READ = 1 << 2
        IBV_ACCESS_RELAXED_ORDERING = 1 << 13
        return (IBV_ACCESS_LOCAL_WRITE
                | IBV_ACCESS_REMOTE_WRITE
                | IBV_ACCESS_REMOTE_READ
                | IBV_ACCESS_RELAXED_ORDERING)


# ---- BAR flush barrier ---------------------------------------------------

def flush_gpudirect_writes(device: Any) -> None:
    """Memory barrier for the consumer side after an RDMA-WRITE lands.

    A NIC completion does NOT establish happens-before with the next CUDA
    kernel that consumes the data; this flush inserts the required
    barrier. Equivalent to ``__threadfence_system`` at the runtime level.
    """
    try:
        import ibverbs.cuda as ibcuda
        with __import__("torch").cuda.device(device):
            ibcuda.flush_gpudirect_writes()
        return
    except Exception:
        pass
    # Fallback: explicit cudaDeviceSynchronize forces all in-flight NIC writes
    # to be visible before subsequent kernels, at the cost of latency.
    try:
        import torch
        torch.cuda.synchronize(device)
    except Exception:
        # Last-resort: ask libcuda for a context-wide sync.
        try:
            _, cudrv = _load_cuda()
            cudrv.cuCtxSynchronize()
        except Exception:
            # No CUDA available (e.g. in CI without GPU). The caller
            # is responsible for ensuring correctness via other means.
            logger.debug("flush_gpudirect_writes: no CUDA runtime available")
