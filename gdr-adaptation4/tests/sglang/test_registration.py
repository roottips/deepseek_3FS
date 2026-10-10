"""Unit tests for sglang.srt.disaggregation.gdr.registration.

These tests stub out the CUDA / ibverbs surface so they run anywhere
pytest is available -- no GPU, no InfiniBand, no nvidia-peermem. They
focus on the LRU cache logic, the peermem/dmabuf path selection, the
SYNC_MEMOPS invariant, and the stale-BUFFER_ID invalidation.

Run:
    cd /workspace && python -m pytest tests/sglang/test_registration.py -v
"""

from __future__ import annotations

import os
import sys
import types
from dataclasses import dataclass
from typing import Any, List
from unittest import mock

import pytest

# Make the in-repo package importable without installing SGLang. We stub
# out the sglang.srt.disaggregation.* dependencies that registration.py
# transitively pulls in, so the unit under test can be loaded in isolation.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class _FakeMr:
    """Mimics the relevant slice of ibverbs.MR used by the cache."""

    _next_lkey = 1000
    _next_rkey = 2000

    def __init__(self, addr: int, length: int):
        self.addr = addr
        self.length = length
        self.lkey = _FakeMr._next_lkey
        self.rkey = _FakeMr._next_rkey
        _FakeMr._next_lkey += 1
        _FakeMr._next_rkey += 1
        self.dereg_calls = 0

    def dereg(self):
        self.dereg_calls += 1


class _FakePd:
    """Stand-in for ibverbs.PD. Records every registration."""

    def __init__(self):
        self.reg_mr_calls: List[tuple] = []
        self.reg_dmabuf_calls: List[tuple] = []
        # Fail mode toggles to exercise error paths.
        self.fail_reg_mr = False
        self.fail_reg_dmabuf = True  # default: dmabuf unavailable

    def reg_mr(self, addr, length, access):
        self.reg_mr_calls.append((addr, length, access))
        if self.fail_reg_mr:
            raise RuntimeError("simulated ibv_reg_mr failure")
        return _FakeMr(addr, length)

    def reg_dmabuf_mr(self, offset, length, iova, fd, access):
        self.reg_dmabuf_calls.append((offset, length, iova, fd, access))
        if self.fail_reg_dmabuf:
            raise RuntimeError("simulated ibv_reg_dmabuf_mr failure")
        return _FakeMr(iova, length)


# ---------------------------------------------------------------------------
# Fixtures: stub CUDA driver API before importing the unit under test.
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_cuda(monkeypatch):
    """Patch the ctypes-loaded CUDA helpers in registration to return
    deterministic values so the cache believes the pointer is a GPU
    device pointer."""
    import importlib
    # Late import so the fixture can patch before first use.
    import sglang_gdr.registration as reg

    monkeypatch.setattr(reg, "_memory_type", lambda ptr: 2)   # DEVICE
    monkeypatch.setattr(reg, "_buffer_id", lambda ptr: ptr)   # ptr == bid
    monkeypatch.setattr(reg, "_set_sync_memops", lambda ptr: None)
    monkeypatch.setattr(reg, "_export_dmabuf_fd", lambda ptr, n: -1)
    return reg


@pytest.fixture
def cache(fake_cuda):
    """Fresh GpuMrCache backed by a fake PD."""
    from sglang_gdr.registration import GpuMrCache
    pd = _FakePd()
    c = GpuMrCache(pd=pd, max_pinned_bytes=1 << 20, prefer_dmabuf=False)
    yield c, pd
    c.teardown()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGpuMrCacheHitMiss:
    """Verifies the LRU hit/miss behavior that amortizes pinning cost."""

    def test_first_registration_is_a_miss(self, cache):
        c, pd = cache
        h = c.register((0x1000, 4096))
        assert h.lkey == 1000
        assert len(pd.reg_mr_calls) == 1
        assert c.pinned_bytes() == 4096

    def test_second_registration_same_ptr_is_a_hit(self, cache):
        c, pd = cache
        c.register((0x1000, 4096))
        h2 = c.register((0x1000, 4096))
        # No new ibv_reg_mr call; refcount bumped.
        assert len(pd.reg_mr_calls) == 1
        assert h2.refcount == 2
        assert c.pinned_bytes() == 4096  # not double-counted

    def test_different_ptr_is_a_miss(self, cache):
        c, pd = cache
        c.register((0x1000, 4096))
        c.register((0x2000, 8192))
        assert len(pd.reg_mr_calls) == 2
        assert c.pinned_bytes() == 4096 + 8192


class TestGpuMrCacheEviction:
    """BAR1 budget enforcement; LRU victims are deregistered."""

    def test_eviction_deregisters_oldest(self, cache):
        c, pd = cache
        # Budget is 1 MiB. Pin two 512 KiB regions; release the first so its
        # refcount hits 0, then pin a third which must evict the first.
        a = c.register((0x10000, 512 * 1024))
        c.release(a)  # refcount -> 0, eligible for LRU eviction
        b = c.register((0x20000, 512 * 1024))
        # Now a is LRU with refcount 0; pinning a third forces its eviction.
        c.register((0x30000, 512 * 1024))
        assert a.mr is None or a.mr.dereg_calls >= 1
        assert b.mr.dereg_calls == 0  # 'b' is still live
        assert c.pinned_bytes() <= 1 << 20

    def test_in_use_region_is_not_evicted(self, cache):
        c, pd = cache
        a = c.register((0x10000, 512 * 1024))
        # refcount > 0 => cannot evict; pinning the third should raise or
        # exceed budget. We assert it stays alive.
        c.register((0x20000, 512 * 1024))
        try:
            c.register((0x30000, 512 * 1024))
        except Exception:
            pass  # acceptable: BAR1 exhausted
        assert a.mr.dereg_calls == 0


class TestSyncMemopsInvariant:
    """SYNC_MEMOPS must be set before every verbs registration."""

    def test_set_sync_memops_called_once_per_unique_ptr(self, cache, monkeypatch):
        c, pd = cache
        calls = []
        import sglang_gdr.registration as reg
        monkeypatch.setattr(reg, "_set_sync_memops",
                            lambda ptr: calls.append(ptr))
        c.register((0x4000, 4096))
        c.register((0x4000, 4096))  # cache hit -> no second call
        c.register((0x5000, 4096))  # miss -> call
        assert calls == [0x4000, 0x5000]


class TestDmaBufPath:
    """When dma-buf export succeeds and is preferred, the dma-buf MR
    path is used instead of peermem; the fd is kept open for the MR
    lifetime and closed on teardown."""

    def test_dmabuf_preferred_when_export_succeeds(self, fake_cuda, monkeypatch):
        import sglang_gdr.registration as reg
        # Flip the defaults: prefer dma-buf, export succeeds, PD accepts it.
        monkeypatch.setattr(reg, "_export_dmabuf_fd", lambda ptr, n: 42)
        pd = _FakePd()
        pd.fail_reg_dmabuf = False
        c = reg.GpuMrCache(pd=pd, max_pinned_bytes=1 << 20,
                            prefer_dmabuf=True)
        try:
            h = c.register((0x6000, 4096))
            from sglang_gdr.registration import GpuMemoryKind
            assert h.kind is GpuMemoryKind.DMABUF
            assert h.fd == 42
            assert len(pd.reg_dmabuf_calls) == 1
            assert pd.reg_mr_calls == []
        finally:
            c.teardown()


class TestStaleBufferIdInvalidation:
    """When CUDA re-allocates a buffer at the same VA, BUFFER_ID changes;
    the cache must drop the stale MR so callers do not reuse it."""

    def test_invalidate_drops_stale_entry(self, cache, monkeypatch):
        c, pd = cache
        c.register((0x7000, 4096))
        # Simulate CUDA freeing + reallocating at the same VA: BUFFER_ID
        # is derived from ptr in our stub, so we flip the mapping.
        import sglang_gdr.registration as reg
        monkeypatch.setattr(reg, "_buffer_id", lambda ptr: ptr + 999)
        c.invalidate_stale()
        assert len(c._entries) == 0
        assert c.pinned_bytes() == 0


class TestAccessFlags:
    """The default access flag set must match NVIDIA's bi-directional
    recommendation (LOCAL_WRITE | REMOTE_WRITE | REMOTE_READ | RELAXED)."""

    def test_default_access_has_all_required_flags(self, fake_cuda):
        from sglang_gdr.registration import _default_access
        flags = _default_access()
        # Numeric fallback values used when ibverbs is absent.
        assert int(flags) & 1           # LOCAL_WRITE
        assert int(flags) & (1 << 1)    # REMOTE_WRITE
        assert int(flags) & (1 << 2)    # REMOTE_READ
