"""GDR KV-cache transfer manager, sender, and receiver.

This is the SGLang-facing API. It mirrors the Mooncake/NIXL backends but
implements the data path with native GPU-direct verbs primitives (no
external transfer-engine dependency).

* :class:`GdrKVManager` - per-rank: owns one :class:`GdrTransport` plus a
  :class:`GpuMrCache`. Registers the KV-cache layers with the verbs layer
  and exposes the resulting ``rkey`` / device VA to peers.
* :class:`GdrKVSender`    - prefill side: tracks which KV indices to ship
  and posts batched RDMA-WRITEs into the decode-side KV-cache.
* :class:`GdrKVReceiver`  - decode side: advertises the destination KV
  MRs (rkey + addr) to the prefill side and reports completion via the
  common ``KVPoll`` state machine.
* :class:`GdrKVBootstrapServer` - optional ZMQ-based metadata server so
  multiple prefill ranks can find the decode rank's bootstrap port.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

try:
    # When installed inside the SGLang tree
    from sglang.srt.disaggregation.base.conn import (
        BaseKVManager,
        BaseKVSender,
        BaseKVReceiver,
        BaseKVBootstrapServer,
        KVArgs,
        KVPoll,
    )
    from sglang.srt.disaggregation.common.utils import (
        AuxDataCodec,
        group_concurrent_contiguous,
    )
    from sglang.srt.disaggregation.common.staging_handler import StagingManagerMixin
    from sglang.srt.disaggregation.utils import (
        DisaggregationMode,
        KVTransferDestination,
    )
    from sglang.srt.disaggregation.gdr.registration import (
        GpuMrCache,
        GpuMrHandle,
        flush_gpudirect_writes,
    )
    from sglang.srt.disaggregation.gdr.transport import (
        GdrTransport,
        GdrTransportConfig,
        PeerInfo,
        PendingWrite,
    )
except (ImportError, ModuleNotFoundError):
    # Standalone layout: sglang framework classes are stubbed/mocked;
    # gdr submodules are local siblings.
    from .registration import (
        GpuMrCache,
        GpuMrHandle,
        flush_gpudirect_writes,
    )
    from .transport import (
        GdrTransport,
        GdrTransportConfig,
        PeerInfo,
        PendingWrite,
    )
    # Framework base classes fall back to stubs (set by the test harness).
    # If no stub is installed, these names will be None at runtime.
    import sys as _sys
    _base = _sys.modules.get("sglang.srt.disaggregation.base.conn")
    if _base is not None:
        BaseKVManager = getattr(_base, "BaseKVManager", object)
        BaseKVSender = getattr(_base, "BaseKVSender", object)
        BaseKVReceiver = getattr(_base, "BaseKVReceiver", object)
        BaseKVBootstrapServer = getattr(_base, "BaseKVBootstrapServer", object)
        KVArgs = getattr(_base, "KVArgs", None)
        KVPoll = getattr(_base, "KVPoll", None)
    else:
        BaseKVManager = BaseKVSender = BaseKVReceiver = BaseKVBootstrapServer = object
        KVArgs = KVPoll = None

    _common_utils = _sys.modules.get("sglang.srt.disaggregation.common.utils")
    if _common_utils is not None:
        AuxDataCodec = getattr(_common_utils, "AuxDataCodec", None)
        group_concurrent_contiguous = getattr(
            _common_utils, "group_concurrent_contiguous", None)
    else:
        AuxDataCodec = None
        group_concurrent_contiguous = None

    _common_stage = _sys.modules.get("sglang.srt.disaggregation.common.staging_handler")
    if _common_stage is not None:
        StagingManagerMixin = getattr(_common_stage, "StagingManagerMixin", object)
    else:
        StagingManagerMixin = object

    _disagg_utils = _sys.modules.get("sglang.srt.disaggregation.utils")
    if _disagg_utils is not None:
        DisaggregationMode = getattr(_disagg_utils, "DisaggregationMode", None)
        KVTransferDestination = getattr(_disagg_utils, "KVTransferDestination", None)
    else:
        DisaggregationMode = None
        KVTransferDestination = None

logger = logging.getLogger(__name__)

# Tunable: WR_ID namespace is uint64. High bits encode layer+slot so the
# completion poller can correlate back to (req, layer, chunk) without a
# hash-table lookup on the hot path.
_WR_ID_LAYER_SHIFT = 32
_WR_ID_COUNTER_MASK = (1 << _WR_ID_LAYER_SHIFT) - 1


@dataclass
class _RemoteMr:
    """Snapshot of a peer's GPU memory region after bootstrap exchange."""

    addr: int
    rkey: int
    nbytes: int
    handle: Optional[Any]  # opaque to SGLang; used by transport internally


@dataclass
class _TransferContext:
    """Per-request sender/receiver bookkeeping."""

    room: int
    # Indices pending issue / pending completion
    pending_chunks: List[Tuple[int, int]] = field(default_factory=list)
    issued: int = 0
    completed: int = 0
    failed: Optional[str] = None
    peer: Optional[PeerInfo] = None
    aux_index: Optional[int] = None
    state_indices: Optional[List[List[int]]] = None


# --------------------------------------------------------------------- manager


class GdrKVManager(StagingManagerMixin, BaseKVManager):
    """Per-rank GDR transport owner.

    Mirrors the lifecycle of ``MooncakeKVManager`` so the existing
    scheduler mixins (prefill/decode) can use it without changes.
    """

    supports_role_switch = True
    supports_deferred_decode_kv_release = True

    def __init__(self, args: KVArgs,
                 disaggregation_mode: DisaggregationMode,
                 server_args,
                 is_mla_backend: Optional[bool] = False):
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)
        self.kv_args = args
        self.disaggregation_mode = disaggregation_mode
        self.server_args = server_args
        self.is_mla_backend = is_mla_backend

        # Build the transport + GPU-MR cache.
        cfg = self._build_transport_config()
        self.cache = GpuMrCache(pd=None)  # PD assigned at open() time
        self.transport = GdrTransport(cfg, self.cache)
        self.transport.open()
        # Now PD is live, re-create the cache with the real PD.
        self.cache = GpuMrCache(pd=self.transport.pd)
        self.transport._mr_cache = self.cache

        # Pin KV-cache layers; rkey/addr become the advertised endpoints.
        self.kv_mrs: List[GpuMrHandle] = []
        self.aux_mrs: List[GpuMrHandle] = []
        self.state_mrs: List[List[GpuMrHandle]] = []
        self._register_buffers()

        # Worker pool for prefill-side batch dispatch.
        self._dispatch_threads: List[threading.Thread] = []
        self._dispatch_queue = None
        self._stop = threading.Event()
        if disaggregation_mode == DisaggregationMode.PREFILL:
            self._init_dispatch_pool()

        # Bootstrap servers (decode side) keyed by room.
        self._bootstrap_servers: Dict[int, Any] = {}

    # ---- config ---------------------------------------------------------

    def _build_transport_config(self) -> GdrTransportConfig:
        import os
        return GdrTransportConfig(
            ib_device=os.environ.get("SGLANG_GDR_IB_DEVICE", "mlx5_0"),
            port=int(os.environ.get("SGLANG_GDR_IB_PORT", "1")),
            gid_index=int(os.environ.get("SGLANG_GDR_GID_INDEX", "3")),
            cq_depth=int(os.environ.get("SGLANG_GDR_CQ_DEPTH", "1024")),
            max_wr_send=int(os.environ.get("SGLANG_GDR_MAX_WR_SEND", "256")),
            signalled_every=int(os.environ.get("SGLANG_GDR_SIGNAL_EVERY", "64")),
            use_imm_notify=os.environ.get(
                "SGLANG_GDR_IMM_NOTIFY", "1") == "1",
        )

    # ---- buffer registration -------------------------------------------

    def _registerable_regions(self) -> List[Tuple[Any, int]]:
        """Collect (tensor, nbytes) pairs that must be pinned for RDMA.

        Deduplicates by ``(ptr, nbytes)`` because many KV layers may alias
        the same physical pool (MLA / unified-memory layouts).
        """
        seen_ptrs: set = set()
        regions: List[Tuple[Any, int]] = []
        # KV layers come as raw (ptr, len) pairs from the scheduler (already
        # a CUDA tensor internally). Wrap them as opaque handles for cache.
        a = self.kv_args
        for ptr, length in zip(a.kv_data_ptrs, a.kv_data_lens):
            if ptr in seen_ptrs:
                continue
            seen_ptrs.add(ptr)
            regions.append((ptr, length))
        for ptr, length in zip(a.aux_data_ptrs, a.aux_data_lens):
            if ptr in seen_ptrs:
                continue
            seen_ptrs.add(ptr)
            regions.append((ptr, length))
        for ptrs, lens in zip(a.state_data_ptrs, a.state_data_lens):
            for ptr, length in zip(ptrs, lens):
                if ptr in seen_ptrs:
                    continue
                seen_ptrs.add(ptr)
                regions.append((ptr, length))
        return regions

    def _register_buffers(self) -> None:
        """Pin all KV-cache regions once; cache keeps MRs alive."""
        regions = self._registerable_regions()
        a = self.kv_args
        # Note: the kv_data_ptrs passed in may be raw int addresses (the
        # KV-cache is a torch tensor in sgl; the manager receives ints via
        # ``kv_data_ptrs``). For pinning we reconstruct (ptr, nbytes) pairs
        # which is what GpuMrCache.register accepts.
        self.kv_mrs = [
            self.cache.register((ptr, length))
            for ptr, length in zip(a.kv_data_ptrs, a.kv_data_lens)
        ]
        self.aux_mrs = [
            self.cache.register((ptr, length))
            for ptr, length in zip(a.aux_data_ptrs, a.aux_data_lens)
        ]
        self.state_mrs = [
            [self.cache.register((ptr, length)) for ptr, length in zip(ptrs, lens)]
            for ptrs, lens in zip(a.state_data_ptrs, a.state_data_lens)
        ]
        logger.info(
            "GdrKVManager: pinned %d KV layers (%.2f GiB), %d aux regions, "
            "%d state groups",
            len(self.kv_mrs),
            sum(m.nbytes for m in self.kv_mrs) / (1 << 30),
            len(self.aux_mrs), len(self.state_mrs))

    def _deregister_buffers(self) -> None:
        # GpuMrCache.teardown() drops every MR atomically.
        self.cache.teardown()
        self.kv_mrs.clear()
        self.aux_mrs.clear()
        self.state_mrs.clear()

    # ---- dispatch pool (prefill side) -----------------------------------

    def _init_dispatch_pool(self) -> None:
        from queue import Queue
        n = max(1, int(getattr(self.server_args, "gdr_worker_threads", 4)))
        self._dispatch_queue: Queue = Queue()
        for i in range(n):
            t = threading.Thread(target=self._dispatch_worker,
                                  name=f"gdr-dispatch-{i}", daemon=True)
            t.start()
            self._dispatch_threads.append(t)

    def _dispatch_worker(self) -> None:
        from queue import Empty
        while not self._stop.is_set():
            try:
                task = self._dispatch_queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                self._dispatch_transfer(*task)
            except Exception as exc:
                logger.exception("dispatch worker error: %s", exc)
                task_ctx: _TransferContext = task[3]
                task_ctx.failed = str(exc)

    def _dispatch_transfer(self, conn, src_handle: GpuMrHandle,
                           src_indices: npt.NDArray[np.int32],
                           peer: PeerInfo,
                           ctx: _TransferContext) -> None:
        """Post one layer's worth of contiguous index runs as RDMA-WRITEs.

        Uses ``group_concurrent_contiguous`` (shared util) so a run like
        [7,8,9,12,13] becomes 2 WRs of length 3 and 2.
        """
        # Compute byte offsets: each kv slot is (item_len * page_size) bytes.
        item_len = self.kv_args.kv_item_lens[0]  # bytes per token per layer
        page_size = self.kv_args.page_size
        slot_bytes = item_len * page_size

        for chunk in group_concurrent_contiguous(src_indices):
            start_idx = int(chunk[0])
            count = len(chunk)
            src_offset = start_idx * slot_bytes
            dst_offset = start_idx * slot_bytes
            length = count * slot_bytes
            wr_id = (ctx.room << 8) | ctx.issued  # bounded per request
            self.transport.completion_queue(ctx.room)  # ensure queue exists
            conn.post_write(PendingWrite(
                src_handle_mr=src_handle,
                src_offset=src_offset,
                peer_info=peer,
                dst_offset=dst_offset,
                length=length,
                wr_id=wr_id,
                signalled=(ctx.issued % self.transport._config.signalled_every == 0),
            ))
            ctx.issued += 1
        # Ensure all WRs that depend on the producing CUDA kernel are visible.
        flush_gpudirect_writes(self.kv_args.gpu_id)

    # ---- BaseKVManager overrides ----------------------------------------

    def register_to_bootstrap(self) -> None:
        # Decode side: register advertised KV MRs to the bootstrap server so
        # the prefill rank can resolve peer rkey/addr/port.
        # Implementation mirrors MooncakeKVManager.register_to_bootstrap.
        bootstrap_addr = self.kv_args.bootstrap_addr  # set by scheduler
        self._bootstrap_server = self._create_bootstrap_server(bootstrap_addr)

    def _create_bootstrap_server(self, bootstrap_addr: str):
        # Reuse the common ZMQ-based bootstrap if available; otherwise a
        # simple in-process stub is sufficient for single-node tests.
        try:
            from sglang.srt.disaggregation.common.bootstrap import (
                DeferredBootstrap,
            )
            return DeferredBootstrap(bootstrap_addr)
        except Exception:
            return None

    def teardown(self) -> None:
        self._stop.set()
        self._deregister_buffers()
        for s in list(self._bootstrap_servers.values()):
            try:
                s.close()
            except Exception:
                pass
        self._bootstrap_servers.clear()
        self.transport.close()


# ---------------------------------------------------------------------- sender


class GdrKVSender(BaseKVSender):
    """Prefill side: ship KV-cache + state to the decode rank."""

    def __init__(self, mgr: GdrKVManager, bootstrap_addr: str,
                 bootstrap_room: int,
                 req_has_disagg_prefill_dp_rank: bool = False):
        self.mgr = mgr
        self.bootstrap_addr = bootstrap_addr
        self.bootstrap_room = bootstrap_room
        self.req_has_disagg_prefill_dp_rank = req_has_disagg_prefill_dp_rank
        self.ctx = _TransferContext(room=bootstrap_room)
        self.peer: Optional[PeerInfo] = None
        self.connection = None

    # ---- lifecycle ------------------------------------------------------

    def init(self, num_kv_indices: int,
             aux_index: Optional[int] = None) -> None:
        """Allocate a room, open QP, advertise our src MR.

        The decode side does the symmetric step in
        :meth:`GdrKVReceiver.init`; both ends rendezvous over TCP.
        """
        a = self.mgr.kv_args
        local_peer = PeerInfo(
            lid=0, gid=[0]*16, qpn=0, psn=0,
            peer_addr=int(self.mgr.kv_mrs[0].addr),
            peer_rkey=int(self.mgr.kv_mrs[0].rkey),
            peer_nbytes=int(self.mgr.kv_mrs[0].nbytes),
            room=self.bootstrap_room)
        # Decode rank's bootstrap TCP port lives on the bootstrap addr.
        # Format: "host:port"; host=="" => same as bootstrap_addr host.
        host = self.bootstrap_addr.split(":")[0]
        port = self.mgr.transport._config.bootstrap_port + (
            self.bootstrap_room % 1000)
        self.connection = self.mgr.transport.connect(
            self.bootstrap_room, host, port, local_peer)
        self.peer = self.connection._peer
        self.ctx.peer = self.peer
        self.ctx.aux_index = aux_index

    def send(self, kv_indices: npt.NDArray[np.int32],
             state_indices: Optional[List] = None,
             num_kv_tokens: Optional[int] = None) -> None:
        """Queue one KV-cache layer's indices for dispatch.

        Called once per layer by the scheduler mixin; we batch across
        layers transparently because all layers share the same QP.
        """
        assert self.connection is not None, "init() not called"
        # Each layer has its own MR; we dispatch one task per layer.
        for layer_id, src_handle in enumerate(self.mgr.kv_mrs):
            self.mgr._dispatch_queue.put(
                (self.connection, src_handle, kv_indices,
                 self.peer, self.ctx))
        if state_indices is not None:
            for state_group_id, (mrs, idx_list) in enumerate(
                    zip(self.mgr.state_mrs, state_indices)):
                for layer_id, src_handle in enumerate(mrs):
                    self.mgr._dispatch_queue.put(
                        (self.connection, src_handle,
                         np.asarray(idx_list, dtype=np.int32),
                         self.peer, self.ctx))
        self.ctx.state_indices = state_indices

    def mark_prefill_complete(self) -> None:
        """No-op for GDR: data path is fire-and-forget RDMA-WRITE; the
        decode side observes completion via ``poll()``.
        """
        # Flush any straggler CUDA work to BAR1 before we declare done.
        flush_gpudirect_writes(self.mgr.kv_args.gpu_id)

    def get_max_transfer_tokens(self) -> Optional[int]:
        # GDR is bounded by QP WQE depth, not by page-pool chunks.
        return None

    def get_transfer_metric(self) -> Tuple[int, int]:
        """Return (bytes_transferred, transfer_time_ms)."""
        return (self.ctx.completed * self.mgr.kv_args.kv_item_lens[0]
                * self.mgr.kv_args.page_size, 0)

    def poll(self) -> KVPoll:
        """Drain peer's CQE queue; transition state when all done."""
        if self.ctx.failed is not None:
            return KVPoll.Failed
        if self.connection is None:
            return KVPoll.Bootstrapping
        q = self.mgr.transport.completion_queue(self.ctx.room)
        while True:
            try:
                wr_id, ok, status = q.get_nowait()
            except Exception:
                break
            if not ok:
                self.ctx.failed = status
                return KVPoll.Failed
            self.ctx.completed += 1
        if self.ctx.issued == 0:
            return KVPoll.WaitingForInput
        if self.ctx.completed >= self.ctx.issued:
            return KVPoll.Success
        return KVPoll.Transferring

    def failure_exception(self):
        if self.ctx.failed is None:
            return None
        raise RuntimeError(f"GDR transfer failed for room "
                           f"{self.ctx.room}: {self.ctx.failed}")

    def clear(self) -> None:
        self.ctx = _TransferContext(room=self.bootstrap_room)
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def abort(self) -> None:
        self.ctx.failed = "aborted by caller"
        if self.connection is not None:
            self.connection.close()
            self.connection = None


# -------------------------------------------------------------------- receiver


class GdrKVReceiver(BaseKVReceiver):
    """Decode side: advertise destination MRs, observe inbound writes."""

    def __init__(self, mgr: GdrKVManager, bootstrap_addr: str,
                 bootstrap_room: Optional[int] = None):
        self.mgr = mgr
        self.bootstrap_addr = bootstrap_addr
        self.bootstrap_room = bootstrap_room or 0
        self.ctx = _TransferContext(room=self.bootstrap_room)
        self.connection = None
        self._server = None

    def init(self, prefill_dp_rank: int) -> None:
        """Open a bootstrap TCP listener for our room; the prefill rank
        connects, exchanges peer-info, and both QPs reach RTS.
        """
        a = self.mgr.kv_args
        local_peer = PeerInfo(
            lid=0, gid=[0]*16, qpn=0, psn=0,
            peer_addr=int(self.mgr.kv_mrs[0].addr),
            peer_rkey=int(self.mgr.kv_mrs[0].rkey),
            peer_nbytes=int(self.mgr.kv_mrs[0].nbytes),
            room=self.bootstrap_room)
        self._server = self.mgr.transport.create_server(self.bootstrap_room)
        # Block until the prefill side connects; in practice the scheduler
        # spins poll() so we cannot block here, defer accept to a thread.
        threading.Thread(target=self._accept_async,
                         args=(local_peer,), daemon=True).start()

    def _accept_async(self, local_peer: PeerInfo) -> None:
        try:
            sock = self._server.take_pending_sock(timeout=60)
            self.connection = self.mgr.transport.accept(
                self.bootstrap_room, sock, local_peer)
        except Exception as exc:
            self.ctx.failed = str(exc)

    def send_metadata(self, kv_indices: npt.NDArray[np.int32],
                      aux_index: Optional[int] = None,
                      state_indices: Optional[List] = None,
                      decode_prefix_len: Optional[int] = None,
                      destination: KVTransferDestination = (
                          KVTransferDestination.DEVICE)) -> None:
        """Stash the indices we expect to receive.

        On the decode side the peer pushes data into our KV-cache slots
        via RDMA-WRITE; we just track which indices should appear.
        """
        if destination != KVTransferDestination.DEVICE:
            raise NotImplementedError(
                "GDR backend only supports DEVICE destination; "
                f"requested {destination}")
        # Determine the expected count of completed WRs.
        # We expect one completion (RDMA-WRITE-WITH-IMM) per layer per
        # contiguous chunk sent by the prefill side; exact count is known
        # only after the peer issues them, so we track via CQE arrivals.
        self.ctx.pending_chunks = group_concurrent_contiguous(kv_indices)
        self.ctx.aux_index = aux_index
        self.ctx.state_indices = state_indices

    def poll(self) -> KVPoll:
        if self.ctx.failed is not None:
            return KVPoll.Failed
        if self.connection is None:
            return KVPoll.Bootstrapping
        if not self.connection.ready(timeout=0):
            return KVPoll.Bootstrapping
        # Drain CQEs. On the decode side, RDMA-WRITE-WITH-IMM produces a
        # receive CQE that we use to count completed chunks.
        q = self.mgr.transport.completion_queue(self.ctx.room)
        while True:
            try:
                wr_id, ok, status = q.get_nowait()
            except Exception:
                break
            if not ok:
                self.ctx.failed = status
                return KVPoll.Failed
            self.ctx.completed += 1
        # We are 'done' when we have observed at least one CQE for every
        # chunk we advertised; since the IMM carries the peer's wr_id
        # which encodes layer+counter, the simplest correctness check is
        # 'completed == len(pending_chunks) * num_layers'.
        num_layers = max(1, len(self.mgr.kv_mrs))
        expected = len(self.ctx.pending_chunks) * num_layers
        if self.ctx.completed >= expected and self.ctx.completed > 0:
            # Flush NIC writes to be visible to the consumer CUDA kernel.
            flush_gpudirect_writes(self.mgr.kv_args.gpu_id)
            return KVPoll.Success
        return KVPoll.Transferring

    def failure_exception(self):
        if self.ctx.failed is None:
            return None
        raise RuntimeError(f"GDR receive failed for room "
                           f"{self.ctx.room}: {self.ctx.failed}")


# ----------------------------------------------------------- bootstrap server


class GdrKVBootstrapServer(BaseKVBootstrapServer):
    """Topology-aware bootstrap aggregator.

    SGLang's existing ZMQ bootstrap server (per Mooncake/NIXL) carries
    arbitrary JSON payloads; we extend it with the GDR-specific fields
    (peer_addr, peer_rkey, bootstrap TCP port) by reusing the common
    DeferredBootstrap plumbing. The actual PeerInfo handshake happens
    over the dedicated TCP side-channel in :class:`GdrTransport`.
    """

    PAYLOAD_FIELDS = ("bootstrap_tcp_port", "peer_addr", "peer_rkey",
                      "peer_nbytes", "room", "ib_device")

    def __init__(self, port: int = 29999):
        self.port = port
        # We defer to the shared bootstrap server from common.bootstrap
        # so existing test infrastructure works unchanged.
        try:
            from sglang.srt.disaggregation.common.bootstrap import (
                DeferredBootstrap,
            )
            self._impl = DeferredBootstrap(port=port)
        except Exception:
            self._impl = None

    def register(self, room: int, payload: Dict[str, Any]) -> None:
        if self._impl is None:
            return
        self._impl.register(room, payload)

    def lookup(self, room: int, timeout: float = 60) -> Dict[str, Any]:
        if self._impl is None:
            return {}
        return self._impl.lookup(room, timeout=timeout)

    def close(self) -> None:
        if self._impl is not None:
            self._impl.close()
