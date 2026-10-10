#!/bin/bash
# =========================================================================
# GDR (GPU Direct RDMA) environment bootstrap for SGLang + 3FS.
#
# Source this file once per node before launching SGLang servers or
# 3FS clients. It validates the prerequisites (kernel module, IB device,
# CUDA) and exports the runtime tunables consumed by the modules
# implemented in sglang_gdr/ and threefs_gdr/.
#
# Usage:
#   source /workspace/config/gdr_env.sh
#   python -m sglang.launch_server --disaggregation-transfer-engine gdr ...
# =========================================================================

set -e

# ---- 1. GPUDirect RDMA kernel module (legacy path) ---------------------
# nvidia-peermem ships with CUDA >= 11.4 (R470+ driver). Loaded once per
# host; must be done by an operator with CAP_SYS_MODULE.
#
# When CUDA >= 13 is available, the dma-buf path does NOT need this
# module; we still load it as a fallback for older stacks.
if ! lsmod 2>/dev/null | grep -q '^nvidia_peermem'; then
    if [ "$(id -u)" -eq 0 ]; then
        modprobe nvidia-peermem || echo "[gdr_env] WARN: nvidia-peermem load failed; relying on dma-buf path"
    else
        echo "[gdr_env] WARN: nvidia-peermem not loaded and you are not root."
        echo "         Run: sudo modprobe nvidia-peermem"
        echo "         Or upgrade to CUDA 13+ for the dma-buf path."
    fi
fi

# ---- 2. PyTorch CUDA VMM backend (required for dma-buf export) ---------
# This MUST be set before torch import. Without expandable_segments the
# cuMemGetHandleForAddressRange() call returns CUDA_ERROR_INVALID_VALUE.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ---- 3. SGLang GDR transport knobs --------------------------------------
# Defaults chosen for H100 + ConnectX-7. Tune per node.
export SGLANG_GDR_IB_DEVICE="${SGLANG_GDR_IB_DEVICE:-mlx5_0}"
export SGLANG_GDR_IB_PORT="${SGLANG_GDR_IB_PORT:-1}"
export SGLANG_GDR_GID_INDEX="${SGLANG_GDR_GID_INDEX:-3}"   # RoCEv2
export SGLANG_GDR_CQ_DEPTH="${SGLANG_GDR_CQ_DEPTH:-1024}"
export SGLANG_GDR_MAX_WR_SEND="${SGLANG_GDR_MAX_WR_SEND:-256}"
export SGLANG_GDR_SIGNAL_EVERY="${SGLANG_GDR_SIGNAL_EVERY:-64}"
export SGLANG_GDR_IMM_NOTIFY="${SGLANG_GDR_IMM_NOTIFY:-1}"
# 0 = prefer peermem path; 1 = prefer dma-buf (CUDA 13+, no kernel module)
export SGLANG_GDR_PREFER_DMABUF="${SGLANG_GDR_PREFER_DMABUF:-0}"

# ---- 4. 3FS USRBIO GPU-buffer knob --------------------------------------
# When 1, the FUSE client's iov ring is allocated via cuMemAlloc and
# registered with each IB device's GpuMemMgr. Storage servers' RDMA-WRITE
# operations then land directly in GPU memory (GDS-over-RDMA).
export HF3FS_IOV_USE_GPU_MEMORY="${HF3FS_IOV_USE_GPU_MEMORY:-1}"
export HF3FS_IOV_GPU_DEVICE_ID="${HF3FS_IOV_GPU_DEVICE_ID:-0}"
# BAR1 budget for the GPU MR cache; default 8 GiB. Lower for multi-tenant.
export HF3FS_GDR_BAR1_BUDGET_BYTES="${HF3FS_GDR_BAR1_BUDGET_BYTES:-$((8 * 1024 * 1024 * 1024))}"

# ---- 5. Prerequisite sanity check --------------------------------------
if ! command -v python3 >/dev/null 2>&1; then
    echo "[gdr_env] ERROR: python3 not found"; exit 1
fi
if ! python3 -c "import ibverbs" 2>/dev/null; then
    echo "[gdr_env] WARN: ibverbs Python package not installed."
    echo "         Install with: pip install ibverbs"
fi
if ! ls /dev/infiniband/ 2>/dev/null | grep -q uverbs; then
    echo "[gdr_env] WARN: no /dev/infiniband/uverbs* devices found."
    echo "         GDR requires an InfiniBand/RoCE HCA (e.g. mlx5_0)."
fi

echo "[gdr_env] ready: IB=${SGLANG_GDR_IB_DEVICE}, GID idx=${SGLANG_GDR_GID_INDEX},"
echo "           3FS iov GPU-backed=${HF3FS_IOV_USE_GPU_MEMORY}, device=${HF3FS_IOV_GPU_DEVICE_ID}"
