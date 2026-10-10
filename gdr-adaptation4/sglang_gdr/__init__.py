"""GPUDirect RDMA (GDR) KV-cache transfer backend for SGLang.

This module implements a zero-copy KV-cache transfer path that lets RDMA
NICs read/write GPU device memory directly, bypassing host staging buffers.
It plugs into SGLang's disaggregation framework as a new transport backend
sibling to Mooncake / NIXL.

Activation (server_args.disaggregation_transfer_engine = "gdr"):
    - Prefill side: ``GdrKVSender`` dispatches RDMA-WRITE batches whose
      source GPU MR is the live KV-cache tensor.
    - Decode side: ``GdrKVReceiver`` exposes the destination KV-cache MR
      (rkey + GPU virtual address) over the bootstrap channel; data is pushed
      into GPU memory by remote NICs without any host copy.

The 3FS counterpart (``threefs_gdr/``) reuses the same verbs/GDR primitives
to let file-system RDMA-WRITE land directly into GPU memory (GDS-over-RDMA),
so the same cache + flush helpers are shared.
"""

try:
    # When installed inside the SGLang tree (python/sglang/srt/disaggregation/gdr/)
    from sglang.srt.disaggregation.gdr.conn import (
        GdrKVManager,
        GdrKVSender,
        GdrKVReceiver,
        GdrKVBootstrapServer,
    )
    from sglang.srt.disaggregation.gdr.registration import (
        GpuMrCache,
        GpuMrHandle,
        GpuMemoryKind,
    )
    from sglang.srt.disaggregation.gdr.transport import (
        GdrTransport,
        GdrTransportConfig,
        GdrConnection,
    )
except (ImportError, ModuleNotFoundError):
    # Standalone package layout (e.g. during development / testing)
    from .conn import (
        GdrKVManager,
        GdrKVSender,
        GdrKVReceiver,
        GdrKVBootstrapServer,
    )
    from .registration import (
        GpuMrCache,
        GpuMrHandle,
        GpuMemoryKind,
    )
    from .transport import (
        GdrTransport,
        GdrTransportConfig,
        GdrConnection,
    )

__all__ = [
    "GdrKVManager",
    "GdrKVSender",
    "GdrKVReceiver",
    "GdrKVBootstrapServer",
    "GpuMrCache",
    "GpuMrHandle",
    "GpuMemoryKind",
    "GdrTransport",
    "GdrTransportConfig",
    "GdrConnection",
]
