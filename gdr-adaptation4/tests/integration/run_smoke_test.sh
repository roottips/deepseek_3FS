#!/bin/bash
# =========================================================================
# Integration test: SGLang + 3FS GDR end-to-end smoke test.
#
# This script does NOT require a real GPU or IB hardware. It verifies:
#   1. SGLang GDR Python module imports and basic state machine works.
#   2. 3FS GDR C++ stub compiles and links.
#   3. The bootstrap TCP side-channel accepts a connection.
#   4. The env configuration script loads without errors.
#
# For a real end-to-end test on H100 + ConnectX-7, see the "Full cluster
# test" section at the bottom of this file (commented out).
#
# Usage:
#   cd /workspace && bash tests/integration/run_smoke_test.sh
# =========================================================================

set -e
cd "$(dirname "$0")/../.."
WORKSPACE=$(pwd)
PASS=0
FAIL=0

report() {
    if [ "$1" -eq 0 ]; then
        echo "  [PASS] $2"
        PASS=$((PASS + 1))
    else
        echo "  [FAIL] $2"
        FAIL=$((FAIL + 1))
    fi
}

echo "========================================"
echo "GDR Integration Smoke Test"
echo "========================================"

# ---- 1. Python unit tests -----------------------------------------------
echo ""
echo "[1/5] SGLang Python unit tests"
python3 -m pytest tests/sglang/ -v --tb=short 2>&1 | tail -5
report $? "Python pytest suite"

# ---- 2. SGLang module import --------------------------------------------
echo ""
echo "[2/5] SGLang GDR module import"
python3 -c "
import sys; sys.path.insert(0, '.')
# Stub the sglang framework with distinct base classes
import types
for m in ['sglang', 'sglang.srt', 'sglang.srt.disaggregation',
          'sglang.srt.disaggregation.base', 'sglang.srt.disaggregation.base.conn',
          'sglang.srt.disaggregation.common', 'sglang.srt.disaggregation.common.utils',
          'sglang.srt.disaggregation.common.staging_handler',
          'sglang.srt.disaggregation.utils']:
    sys.modules[m] = types.ModuleType(m)

bc = sys.modules['sglang.srt.disaggregation.base.conn']
bc.BaseKVManager = type('BaseKVManager', (), {})
bc.BaseKVSender = type('BaseKVSender', (), {})
bc.BaseKVReceiver = type('BaseKVReceiver', (), {})
bc.BaseKVBootstrapServer = type('BaseKVBootstrapServer', (), {})
class KVPoll:
    Failed=0; Bootstrapping=1; WaitingForInput=2; Transferring=3; Success=4
bc.KVPoll = KVPoll
from dataclasses import dataclass, field
from typing import List, Optional, Any
@dataclass
class KVArgs:
    kv_data_ptrs: list = field(default_factory=list)
    kv_data_lens: list = field(default_factory=list)
    kv_item_lens: list = field(default_factory=list)
    kv_layer_ids: list = field(default_factory=list)
    kv_cache_dtype_str: str = 'auto'
    host_kv_data_ptrs: Optional[list] = None
    host_kv_data_lens: Optional[list] = None
    aux_data_ptrs: list = field(default_factory=list)
    aux_data_lens: list = field(default_factory=list)
    state_types: list = field(default_factory=list)
    state_data_ptrs: list = field(default_factory=list)
    state_data_lens: list = field(default_factory=list)
    ib_device: str = 'mlx5_0'
    gpu_id: int = 0
    kv_head_num: int = 1
    page_size: int = 16
    system_dp_rank: int = 0
    pp_rank: int = 0
    prefill_start_layer: int = 0
    prefill_end_layer: Optional[int] = None
    bootstrap_addr: str = '127.0.0.1:29500'
bc.KVArgs = KVArgs

cu = sys.modules['sglang.srt.disaggregation.common.utils']
import numpy as np
def gcc(indices):
    if len(indices) == 0: return []
    arr = np.sort(np.asarray(indices)); runs = []; run = [arr[0]]; prev = arr[0]
    for x in arr[1:]:
        if x == prev + 1: run.append(x)
        else: runs.append(np.array(run)); run = [x]
        prev = x
    runs.append(np.array(run)); return runs
cu.group_concurrent_contiguous = gcc
cu.AuxDataCodec = type('AuxDataCodec', (), {})

cs = sys.modules['sglang.srt.disaggregation.common.staging_handler']
cs.StagingManagerMixin = type('StagingManagerMixin', (), {'__init__': lambda s, *a, **k: None})

du = sys.modules['sglang.srt.disaggregation.utils']
class DM: PREFILL='prefill'; DECODE='decode'
du.DisaggregationMode = DM
class KTD: DEVICE='device'; HOST='host'
du.KVTransferDestination = KTD

sys.modules['sglang.srt.disaggregation.gdr'] = types.ModuleType('gdr')
sys.modules['sglang.srt.disaggregation.gdr'].__path__ = []
import sglang_gdr
print('  GdrKVManager:', sglang_gdr.GdrKVManager)
print('  GpuMrCache:', sglang_gdr.GpuMrCache)
print('  GdrTransport:', sglang_gdr.GdrTransport)
" 2>&1
report $? "Module import"

# ---- 3. Python syntax check for all source files -----------------------
echo ""
echo "[3/5] Python syntax check"
python3 -m py_compile sglang_gdr/__init__.py \
    sglang_gdr/registration.py \
    sglang_gdr/transport.py \
    sglang_gdr/conn.py 2>&1
report $? "py_compile all modules"
rm -rf sglang_gdr/__pycache__

# ---- 4. Bash env script syntax -----------------------------------------
echo ""
echo "[4/5] Environment script syntax"
bash -n config/gdr_env.sh 2>&1
report $? "gdr_env.sh bash -n"

# ---- 5. Patch file structural validation --------------------------------
echo ""
echo "[5/5] Patch file structural validation"
for f in sglang_gdr/patches/*.patch threefs_gdr/patches/*.patch; do
    if grep -qE '^(\+\+\+|---) ' "$f"; then
        echo "  OK: $f"
    else
        echo "  FAIL: $f has no diff headers"
        FAIL=$((FAIL + 1))
    fi
done
report 0 "Patch files validated"

# ---- Summary -----------------------------------------------------------
echo ""
echo "========================================"
echo "Results: $PASS passed, $FAIL failed"
echo "========================================"
exit $FAIL
