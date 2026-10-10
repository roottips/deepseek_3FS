"""GDR transport: verbs QP/CQ management and RDMA-WRITE batching.

Owns one RC QP per (peer, device) pair and multiplexes SGLang's
KV-cache layers over it. This is intentionally lighter than Mooncake /
NIXL: there is no per-segment session object because GPU memory is
pinned once (in :class:`GpuMrCache`) and addressed by ``remote_addr +
rkey`` directly.

Connection bootstrap mirrors 3FS's IBConnectService: a TCP side-channel
exchanges ``{LID, GID, QPN, PSN, peer_rkey, peer_addr, peer_nbytes}``
then both endpoints transition QP RESET -> INIT -> RTR -> RTS.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Poll CQ budget per loop iteration; balances latency vs syscall overhead.
_CQ_POLL_BUDGET = 64
# After this many idle polls, yield via Queue.get(timeout) to save CPU.
_IDLE_POLLS_BEFORE_BLOCK = 4096


@dataclass
class GdrTransportConfig:
    """User-tunable knobs; defaults chosen for H100 + CX-7 InfiniBand."""

    ib_device: str = "mlx5_0"        # /dev/infiniband device basename
    port: int = 1                    # IB port number (1-based)
    gid_index: int = 3               # RoCEv2 default; pure-IB uses 0
    max_sge: int = 8                 # SGE per WR; CX-7 supports up to 128
    max_wr_send: int = 256           # send-WQ depth
    max_wr_recv: int = 64            # recv-WQ depth (only for IMM path)
    cq_depth: int = 1024
    signalled_every: int = 64        # burst a signalled CQE every N WRs
    mtu: int = 1024                  # IB MTU enum: 1024 = IBV_MTU_1024
    timeout: int = 14                # QP_PATH_MTU / 4.096us * 2**timeout
    retry_cnt: int = 7              # per-NIC retry limit before erroring QP
    rnr_retry: int = 7               # RNR retry count
    min_rnr_timer: int = 12          # RNR timer (0.512ms)
    use_imm_notify: bool = True      # emit immediate-data on RDMA-WRITE so
                                     # the receiver gets a CQE on completion
    bootstrap_port: int = 29500      # TCP side-channel port range start


@dataclass
class PeerInfo:
    """Wire-format exchanged over TCP side-channel."""

    lid: int
    gid: List[int]   # 16-byte GID
    qpn: int
    psn: int
    # peer GPU memory region advertised for RDMA-WRITE targeting
    peer_addr: int
    peer_rkey: int
    peer_nbytes: int
    room: int       # SGLang bootstrap room; correlation id

    def to_json(self) -> str:
        return json.dumps({
            "lid": self.lid, "gid": self.gid, "qpn": self.qpn,
            "psn": self.psn, "peer_addr": self.peer_addr,
            "peer_rkey": self.peer_rkey, "peer_nbytes": self.peer_nbytes,
            "room": self.room})

    @classmethod
    def from_json(cls, s: str) -> "PeerInfo":
        d = json.loads(s)
        return cls(lid=d["lid"], gid=d["gid"], qpn=d["qpn"], psn=d["psn"],
                   peer_addr=d["peer_addr"], peer_rkey=d["peer_rkey"],
                   peer_nbytes=d["peer_nbytes"], room=d["room"])


@dataclass
class PendingWrite:
    """Single WR queued to the transport worker."""

    src_handle_mr: Any        # GpuMrHandle (src) -- already pinned
    src_offset: int
    peer_info: PeerInfo
    dst_offset: int          # offset into peer_addr
    length: int
    wr_id: int
    signalled: bool = True


class GdrConnection:
    """One RC QP + its CQ, ready to post RDMA-WRITE into a peer GPU MR."""

    def __init__(self, transport: "GdrTransport", peer: PeerInfo,
                 is_active: bool):
        self._t = transport
        self._peer = peer
        self._is_active = is_active  # active = client side (initiates)
        # ibverbs objects; populated by _establish()
        self._qp: Any = None
        self._cq: Any = None
        self._send_counter: int = 0
        self._pending: Dict[int, PendingWrite] = {}
        self._lock = threading.Lock()
        self._established = threading.Event()

    # ---- lifecycle -------------------------------------------------------

    def establish(self, local_info: PeerInfo, sock: socket.socket) -> None:
        """Two-way exchange over ``sock`` then QP state transitions.

        ``sock`` is a *connected* TCP socket whose handshake is owned by
        :class:`GdrTransport`. We exchange peer-info JSON over it once and
        close it immediately.
        """
        sock.sendall(local_info.to_json().encode() + b"\n")
        peer_json = sock.recv(65536).decode().strip()
        self._peer = PeerInfo.from_json(peer_json)
        sock.close()
        self._qp = self._t._create_qp(self._peer.room)
        self._cq = self._t._cq_for(self._peer.room)
        self._modify_to_init()
        self._modify_to_rtr()
        self._modify_to_rts()
        self._established.set()

    def ready(self, timeout: Optional[float] = None) -> bool:
        return self._established.wait(timeout)

    def close(self) -> None:
        with self._lock:
            if self._qp is not None:
                try:
                    self._qp.destroy()
                except Exception:
                    pass
                self._qp = None

    # ---- QP transitions (verbs) ------------------------------------------

    def _modify_to_init(self) -> None:
        # INIT: set port, access flags, ready to receive REQ.
        try:
            import ibverbs as ib
            self._qp.modify(ib.QPAttr(
                port_num=self._t._config.port,
                qp_access_flags=(ib.AccessFlags.LOCAL_WRITE
                                 | ib.AccessFlags.REMOTE_WRITE
                                 | ib.AccessFlags.REMOTE_READ),
                state=ib.QPState.INIT))
        except Exception as exc:
            logger.error("QP -> INIT failed: %s", exc)
            raise

    def _modify_to_rtr(self) -> None:
        # Ready-to-Receive: program peer's LID/GID/QPN/PSN.
        try:
            import ibverbs as ib
            ah_attr = ib.AHAttr(
                port_num=self._t._config.port,
                dlid=self._peer.lid,
                sgid_index=self._t._config.gid_index,
                gid=tuple(self._peer.gid),
                hop_limit=1, sl=0, traffic_class=0)
            self._qp.modify(ib.QPAttr(
                dest_qp_num=self._peer.qpn,
                ah_attr=ah_attr,
                path_mtu=ib.MTU(self._t._config.mtu),
                rq_psn=self._peer.psn,
                max_dest_rd_atomic=16,
                min_rnr_timer=self._t._config.min_rnr_timer,
                state=ib.QPState.RTR))
        except Exception as exc:
            logger.error("QP -> RTR failed: %s", exc)
            raise

    def _modify_to_rts(self) -> None:
        # Ready-to-Send.
        try:
            import ibverbs as ib
            self._qp.modify(ib.QPAttr(
                sq_psn=self._peer.psn,
                timeout=self._t._config.timeout,
                retry_cnt=self._t._config.retry_cnt,
                rnr_retry=self._t._config.rnr_retry,
                max_rd_atomic=16,
                state=ib.QPState.RTS))
        except Exception as exc:
            logger.error("QP -> RTS failed: %s", exc)
            raise

    # ---- data plane ------------------------------------------------------

    def post_write(self, write: PendingWrite) -> None:
        """Submit one RDMA-WRITE WR (source GPU -> peer GPU).

        Caller is responsible for ensuring the source CUDA work has been
        flushed (see :func:`flush_gpudirect_writes` after polling). The
        NIC reads from the source GPU BAR1 window and writes to the peer
        GPU BAR1 window with zero host involvement.
        """
        assert self._qp is not None, "connection not established"
        try:
            import ibverbs as ib
            sge = ib.SGE(
                addr=write.src_handle_mr.addr + write.src_offset,
                length=write.length,
                lkey=write.src_handle_mr.lkey)
            wr = ib.SendWR(
                wr_id=write.wr_id,
                sg_list=[sge],
                opcode=ib.WROpcode.RDMA_WRITE_WITH_IMM
                       if self._t._config.use_imm_notify
                       else ib.WROpcode.RDMA_WRITE,
                send_flags=ib.SendFlags.SIGNALED
                           if write.signalled else ib.SendFlags(0),
                remote_addr=write.peer_info.peer_addr + write.dst_offset,
                rkey=write.peer_info.peer_rkey,
                immediate_data=write.wr_id
                                if self._t._config.use_imm_notify else None)
            with self._lock:
                bad_wr = ctypes_void()  # placeholder; verbs API below
                self._qp.post_send([wr])
                self._pending[write.wr_id] = write
        except Exception as exc:
            logger.error("post_send RDMA-WRITE failed: %s", exc)
            raise

    def poll_completion(self, max_count: int = _CQ_POLL_BUDGET
                        ) -> List[Tuple[int, bool, str]]:
        """Drain completed WRs. Returns ``[(wr_id, ok, status)]``.

        ``ok=False`` + ``status`` populated when the WC errored (e.g. transport
        retry exhausted); the owning sender must abort the request.
        """
        if self._cq is None:
            return []
        results: List[Tuple[int, bool, str]] = []
        try:
            wcs = self._cq.poll(max_count)
            if not wcs:
                return results
            for wc in wcs:
                wr_id = int(wc.wr_id)
                ok = (wc.status == 0)
                status = "" if ok else str(wc.status)
                results.append((wr_id, ok, status))
                with self._lock:
                    self._pending.pop(wr_id, None)
        except Exception as exc:
            logger.error("poll_cq failed: %s", exc)
        return results


def ctypes_void() -> None:
    return None  # placeholder; the real post_send signature is used above


class GdrTransport:
    """Top-level transport owning ibv_context, PD, CQ pool, and worker.

    A single transport instance per (rank, GPU) serves all bootstrap rooms;
    multiple :class:`GdrConnection` objects multiplex over the same PD and
    one CQ per connection.
    """

    def __init__(self, config: GdrTransportConfig, mr_cache):
        self._config = config
        self._mr_cache = mr_cache
        self._ctx: Any = None
        self._pd: Any = None
        self._ib_device: Any = None
        self._connections: Dict[int, GdrConnection] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        # Per-connection CQ pool; keyed by room.
        self._cqs: Dict[int, Any] = {}
        # Worker: drains CQ completions and forwards them to per-room
        # completion queues so callers can block on a single Queue.
        self._completion_queues: Dict[int, Queue] = {}
        self._worker: Optional[threading.Thread] = None

    # ---- initialization --------------------------------------------------

    def open(self) -> None:
        try:
            import ibverbs as ib
        except ImportError as exc:
            raise RuntimeError(
                "ibverbs package not installed: pip install ibverbs") from exc
        devices = ib.get_device_list()
        target = None
        for dev in devices:
            if dev.name.decode() == self._config.ib_device:
                target = dev
                break
        if target is None:
            raise RuntimeError(f"IB device {self._config.ib_device} not found; "
                               f"available: {[d.name for d in devices]}")
        self._ctx = target.open()
        self._ib_device = self._ctx
        self._pd = self._ctx.alloc_pd()
        # Spawn the CQ poller thread.
        self._worker = threading.Thread(target=self._poll_loop,
                                        name="gdr-poller", daemon=True)
        self._worker.start()

    def close(self) -> None:
        self._stop.set()
        for conn in list(self._connections.values()):
            conn.close()
        self._connections.clear()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None
        if self._mr_cache is not None:
            self._mr_cache.teardown()

    @property
    def pd(self):
        return self._pd

    @property
    def cache(self):
        return self._mr_cache

    # ---- connection establishment ---------------------------------------

    def create_server(self, room: int) -> "GdrBootstrapServer":
        """Decode side opens a TCP listener for the room; the active peer
        connects and exchanges peer-info. Returns a handle to block on.
        """
        return GdrBootstrapServer(self, room)

    def connect(self, room: int, host: str, port: int,
                local_peer: PeerInfo) -> GdrConnection:
        """Prefill side: active TCP dial, exchange info, raise QP to RTS."""
        sock = socket.create_connection((host, port), timeout=30)
        conn = GdrConnection(self, local_peer, is_active=True)
        with self._lock:
            self._connections[room] = conn
            self._completion_queues[room] = Queue()
        conn.establish(local_peer, sock)
        return conn

    def accept(self, room: int, sock: socket.socket,
               local_peer: PeerInfo) -> GdrConnection:
        """Decode side: passive TCP accept was already done by the bootstrap
        server; we now perform the same QP handshake as :meth:`connect`.
        """
        conn = GdrConnection(self, local_peer, is_active=False)
        with self._lock:
            self._connections[room] = conn
            self._completion_queues[room] = Queue()
        conn.establish(local_peer, sock)
        return conn

    def connection(self, room: int) -> Optional[GdrConnection]:
        return self._connections.get(room)

    def completion_queue(self, room: int) -> Queue:
        return self._completion_queues.setdefault(room, Queue())

    # ---- verbs helpers used by GdrConnection ----------------------------

    def _create_qp(self, room: int):
        import ibverbs as ib
        cq = self._cq_for(room)
        return self._pd.create_qp(
            cq_send=cq, cq_recv=cq,
            cap=ib.QPCap(
                max_send_wr=self._config.max_wr_send,
                max_recv_wr=self._config.max_wr_recv,
                max_send_sge=self._config.max_sge,
                max_recv_sge=1,
                max_inline_data=0),
            qp_type=ib.QPType.RC)

    def _cq_for(self, room: int):
        if room not in self._cqs:
            import ibverbs as ib
            self._cqs[room] = self._ctx.create_cq(
                cqe=self._config.cq_depth)
        return self._cqs[room]

    # ---- completion poller ----------------------------------------------

    def _poll_loop(self) -> None:
        idle = 0
        while not self._stop.is_set():
            had_work = False
            for room, conn in list(self._connections.items()):
                if not conn.ready(timeout=0):
                    continue
                wcs = conn.poll_completion()
                if wcs:
                    had_work = True
                    q = self._completion_queues.get(room)
                    if q is not None:
                        for wr_id, ok, status in wcs:
                            q.put((wr_id, ok, status))
            if had_work:
                idle = 0
            else:
                idle += 1
                if idle > _IDLE_POLLS_BEFORE_BLOCK:
                    # Park briefly until new WRs likely arrive.
                    time.sleep(0.0005)
                    idle = 0


class GdrBootstrapServer:
    """Minimal TCP side-channel used to exchange PeerInfo.

    Mirrors 3FS's IBConnectService: the decode side (passive) listens on
    a port derived from ``room``, the prefill side (active) connects, the
    two endpoints swap JSON-serialized peer-info, then both close the
    socket and switch to verbs-only data path.
    """

    def __init__(self, transport: GdrTransport, room: int):
        self._t = transport
        self._room = room
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._port = transport._config.bootstrap_port + (room % 1000)
        self._sock.bind(("0.0.0.0", self._port))
        self._sock.listen(1)
        self._accept_thread = threading.Thread(
            target=self._accept, name=f"gdr-bs-{room}", daemon=True)
        self._accept_thread.start()
        self._accepted = threading.Event()
        self.local_port = self._port

    def _accept(self) -> None:
        try:
            conn, _ = self._sock.accept()
            # The actual handshake is finished by GdrTransport.accept()
            # which receives local_peer; stash the raw socket for it.
            self._pending_sock = conn
            self._accepted.set()
        except Exception as exc:
            logger.error("GdrBootstrapServer accept failed: %s", exc)

    def take_pending_sock(self, timeout: Optional[float] = 30) -> socket.socket:
        self._accepted.wait(timeout=timeout)
        sock = getattr(self, "_pending_sock", None)
        if sock is None:
            raise TimeoutError(f"no bootstrap connection for room {self._room}")
        return sock

    def close(self) -> None:
        try:
            self._sock.close()
        except Exception:
            pass
