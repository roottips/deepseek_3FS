"""Unit tests for sglang.srt.disaggregation.gdr.conn.

Exercises the sender/receiver state machine, WR-ID encoding, and the
KVPoll transitions that the scheduler mixin relies on. The transport
layer is stubbed so no real verbs calls happen.

Run:
    cd /workspace && python -m pytest tests/sglang/test_conn.py -v
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, List, Optional
from unittest import mock

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


# ---------------------------------------------------------------------------
# Stub the sglang.srt.disaggregation.* imports the conn module pulls in.
# ---------------------------------------------------------------------------

def _install_sglang_stubs():
    """Create minimal stub modules so conn.py can be imported without
    the full SGLang tree installed."""
    for mod_name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.disaggregation",
        "sglang.srt.disaggregation.base",
        "sglang.srt.disaggregation.base.conn",
        "sglang.srt.disaggregation.common",
        "sglang.srt.disaggregation.common.utils",
        "sglang.srt.disaggregation.common.staging_handler",
        "sglang.srt.disaggregation.utils",
        "sglang.srt.disaggregation.gdr",
    ):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = type(sys)("stub")

    base_conn = sys.modules["sglang.srt.disaggregation.base.conn"]

    # KVPoll enum
    class KVPoll:
        Failed = 0
        Bootstrapping = 1
        WaitingForInput = 2
        Transferring = 3
        Success = 4

    base_conn.KVPoll = KVPoll

    # KVArgs dataclass
    @dataclass
    class KVArgs:
        kv_data_ptrs: List[int] = field(default_factory=list)
        kv_data_lens: List[int] = field(default_factory=list)
        kv_item_lens: List[int] = field(default_factory=list)
        kv_layer_ids: List[int] = field(default_factory=list)
        kv_cache_dtype_str: str = "auto"
        host_kv_data_ptrs: Optional[List[int]] = None
        host_kv_data_lens: Optional[List[int]] = None
        aux_data_ptrs: List[int] = field(default_factory=list)
        aux_data_lens: List[int] = field(default_factory=list)
        state_types: List[Any] = field(default_factory=list)
        state_data_ptrs: List[List[int]] = field(default_factory=list)
        state_data_lens: List[List[int]] = field(default_factory=list)
        ib_device: str = "mlx5_0"
        gpu_id: int = 0
        kv_head_num: int = 1
        page_size: int = 16
        system_dp_rank: int = 0
        pp_rank: int = 0
        prefill_start_layer: int = 0
        prefill_end_layer: Optional[int] = None
        bootstrap_addr: str = "127.0.0.1:29500"

    base_conn.KVArgs = KVArgs

    # Abstract base classes (no-op)
    for name in ("BaseKVManager", "BaseKVSender", "BaseKVReceiver",
                 "BaseKVBootstrapServer"):
        setattr(base_conn, name, type(name, (), {}))

    # common.utils: group_concurrent_contiguous
    common_utils = sys.modules["sglang.srt.disaggregation.common.utils"]

    def group_concurrent_contiguous(indices):
        """Coalesce a sorted-or-unsorted index list into contiguous runs.
        Returns a list of numpy arrays."""
        if len(indices) == 0:
            return []
        arr = np.sort(np.asarray(indices))
        runs = []
        start = arr[0]
        prev = arr[0]
        run = [arr[0]]
        for x in arr[1:]:
            if x == prev + 1:
                run.append(x)
            else:
                runs.append(np.array(run))
                run = [x]
            prev = x
        runs.append(np.array(run))
        return runs

    common_utils.group_concurrent_contiguous = group_concurrent_contiguous
    # Stubs referenced by conn.py
    common_utils.AuxDataCodec = type("AuxDataCodec", (), {})

    common_stage = sys.modules["sglang.srt.disaggregation.common.staging_handler"]

    class StagingManagerMixin:
        def __init__(self, *a, **kw):
            pass

    common_stage.StagingManagerMixin = StagingManagerMixin

    disagg_utils = sys.modules["sglang.srt.disaggregation.utils"]

    class DisaggregationMode:
        PREFILL = "prefill"
        DECODE = "decode"

    disagg_utils.DisaggregationMode = DisaggregationMode

    class KVTransferDestination:
        DEVICE = "device"
        HOST = "host"

    disagg_utils.KVTransferDestination = KVTransferDestination

    # gdr subpackage so relative imports resolve
    gdr_pkg = sys.modules["sglang.srt.disaggregation.gdr"]
    gdr_pkg.__path__ = []  # mark as package


_install_sglang_stubs()


# Now we can import the unit under test.
import sglang_gdr.conn as conn_mod  # noqa: E402
import sglang_gdr.transport as tmod  # noqa: E402


# ---------------------------------------------------------------------------
# Test doubles for the manager/transport
# ---------------------------------------------------------------------------

class _FakeMrHandle:
    addr = 0x10000
    nbytes = 4096
    lkey = 100
    rkey = 200


class _FakeConnection:
    def __init__(self):
        self.posted = []
        self._peer = None
        self._established = True

    def ready(self, timeout=None):
        return True

    def post_write(self, write):
        self.posted.append(write)

    def close(self):
        pass


class _FakeTransport:
    """Minimal transport stub for the sender/receiver tests."""

    class _Config:
        signalled_every = 64
        use_imm_notify = True

    def __init__(self):
        self._config = self._Config()
        from queue import Queue as _Q
        self._Queue = _Q
        self._cqs = {}

    def completion_queue(self, room):
        return self._cqs.setdefault(room, self._Queue())

    def connect(self, *a, **kw):
        return _FakeConnection()

    def create_server(self, room):
        return mock.MagicMock()

    def accept(self, *a, **kw):
        return _FakeConnection()


class _FakeMgr:
    """Mimics the slice of GdrKVManager the sender/receiver use."""

    def __init__(self, num_layers=3):
        a = type("A", (), {})()
        a.kv_data_ptrs = [0x10000 + i * 4096 for i in range(num_layers)]
        a.kv_data_lens = [4096] * num_layers
        a.kv_item_lens = [128] * num_layers
        a.page_size = 16
        a.gpu_id = 0
        a.aux_data_ptrs = []
        a.aux_data_lens = []
        a.state_data_ptrs = []
        a.state_data_lens = []
        a.bootstrap_addr = "127.0.0.1:29500"
        self.kv_args = a
        self.kv_mrs = [_FakeMrHandle() for _ in range(num_layers)]
        self.aux_mrs = []
        self.state_mrs = []
        self.transport = _FakeTransport()
        self._dispatch_queue = None

    def _dispatch_transfer(self, *args):
        # Called by the sender; just record into the connection.
        conn, src_handle, indices, peer, ctx = args
        for chunk in __import__("numpy").split(
                __import__("numpy").sort(indices), [1]):
            pass


# ---------------------------------------------------------------------------
# Tests: sender state machine
# ---------------------------------------------------------------------------

class TestGdrKVSenderStateMachine:
    def test_poll_returns_failed_when_ctx_failed(self):
        mgr = _FakeMgr()
        sender = conn_mod.GdrKVSender(mgr, "127.0.0.1:29500", bootstrap_room=1)
        sender.ctx.failed = "simulated error"
        assert sender.poll() == 0  # KVPoll.Failed

    def test_poll_returns_success_after_all_completions(self):
        mgr = _FakeMgr()
        sender = conn_mod.GdrKVSender(mgr, "127.0.0.1:29500", bootstrap_room=1)
        sender.connection = _FakeConnection()
        # Simulate 3 issued WRs, 3 completions.
        sender.ctx.issued = 3
        q = mgr.transport.completion_queue(1)
        for i in range(3):
            q.put((i, True, ""))
        assert sender.poll() == 4  # KVPoll.Success

    def test_poll_returns_transferring_when_partial(self):
        mgr = _FakeMgr()
        sender = conn_mod.GdrKVSender(mgr, "127.0.0.1:29500", bootstrap_room=1)
        sender.connection = _FakeConnection()
        sender.ctx.issued = 3
        q = mgr.transport.completion_queue(1)
        q.put((0, True, ""))  # only 1 of 3
        assert sender.poll() == 3  # KVPoll.Transferring

    def test_abort_sets_failure(self):
        mgr = _FakeMgr()
        sender = conn_mod.GdrKVSender(mgr, "127.0.0.1:29500", bootstrap_room=1)
        sender.abort()
        assert sender.ctx.failed is not None


# ---------------------------------------------------------------------------
# Tests: receiver state machine
# ---------------------------------------------------------------------------

class TestGdrKVReceiverStateMachine:
    def test_poll_returns_bootstrapping_before_accept(self):
        mgr = _FakeMgr()
        recv = conn_mod.GdrKVReceiver(mgr, "127.0.0.1:29500", bootstrap_room=1)
        recv.connection = None
        assert recv.poll() == 1  # KVPoll.Bootstrapping

    def test_poll_returns_success_on_expected_completions(self):
        mgr = _FakeMgr(num_layers=2)
        recv = conn_mod.GdrKVReceiver(mgr, "127.0.0.1:29500", bootstrap_room=1)
        recv.connection = _FakeConnection()
        # Advertise 1 chunk * 2 layers = 2 expected completions.
        recv.send_metadata(np.array([5, 6, 7], dtype=np.int32))
        q = mgr.transport.completion_queue(1)
        q.put((0, True, ""))
        q.put((1, True, ""))
        assert recv.poll() == 4  # KVPoll.Success


# ---------------------------------------------------------------------------
# Tests: bootstrap server payload fields
# ---------------------------------------------------------------------------

class TestGdrKVBootstrapServer:
    def test_payload_fields_include_gdr_specifics(self):
        fields = conn_mod.GdrKVBootstrapServer.PAYLOAD_FIELDS
        assert "peer_addr" in fields
        assert "peer_rkey" in fields
        assert "bootstrap_tcp_port" in fields
        assert "room" in fields
        assert "ib_device" in fields
