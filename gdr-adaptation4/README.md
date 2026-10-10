# SGLang + 3FS GPUDirect RDMA (GDR) 适配

**3FS 基线 commit:** `f84e609692dded708f9c910caf5c5af3444e1024` (2025-03-01)

## 概述

本项目为 SGLang 的 KV-Cache 跨节点迁移 和 3FS 的存储 I/O 路径实现了 GPU Direct RDMA (GDR) 模块适配。GDR 让 RDMA NIC 经 BAR1 窗口直接读写 GPU 显存，绕过主机内存，消除传统路径中的两次 `cudaMemcpy`。

## 目录结构

```
gdr-adaptation/
├── sglang_gdr/                     # SGLang 侧 GDR 传输模块 (Python)
│   ├── __init__.py                 # 模块导出 (双路径导入: SGLang树内 / 独立)
│   ├── registration.py              # GPU 内存 LRU pin 缓存 (peermem/dmabuf)
│   ├── transport.py                # verbs QP/CQ 管理 + TCP 侧信道引导
│   ├── conn.py                     # GdrKVManager/Sender/Receiver/BootstrapServer
│   └── patches/
│       ├── parallel_state.py.patch # 暴露 get_gdr_transport() 单例
│       └── utils.py.patch           # 在分发表注册 "gdr" 后端
│
├── threefs_gdr/                    # 3FS 侧 GPU 内存适配 (C++)
│   │                                # 基线: f84e609692dded708f9c910caf5c5af3444e1024
│   ├── include/
│   │   └── GpuMemMgr.h             # GpuVmmAllocation/GpuMrCache/GpuBufferPool/GpuMemMgr (namespace hf3fs::net)
│   ├── src/
│   │   └── GpuMemMgr.cc            # libcuda 弱加载 + VMM API + LRU 淘汰 + stale检测
│   └── patches/
│       ├── IBDevice.h.patch         # 仅添加 bar1Budget() (pd()/name() 已在基线 public)
│       ├── RDMABuf.h.patch          # gpu_handles_ 数组 + gpu_backed_ flag + detectGpuPointer_()
│       ├── RDMABuf.cc.patch         # GPU 指针检测 + 路由到 GpuMemMgr (适配 mrs_ 数组)
│       └── Shm.cc.patch             # mapBufGpu_: VMM API (cuMemCreate/cuMemMap) + Shm.h + IoRing.h
│
├── tests/
│   ├── sglang/                      # Python 单元测试 (pytest)
│   │   ├── test_registration.py     # 9 tests: LRU/淘汰/SYNC_MEMOPS/dmabuf/stale
│   │   ├── test_transport.py        # 7 tests: PeerInfo/config/bootstrap/CQ
│   │   └── test_conn.py             # 7 tests: sender/receiver 状态机
│   ├── threefs/
│   │   └── test_GpuMemMgr.cpp       # C++ 单元测试 (stub IBDevice)
│   └── integration/
│       └── run_smoke_test.sh        # 5 步端到端验证脚本
│
├── config/
│   ├── gdr_env.sh                   # 环境变量 bootstrap 脚本
│   └── CMakeLists.gdr.txt           # CMake 构建配置片段
│
└── README.md                        # 本文件
```

## 架构设计

### 数据通路对比

**传统路径 (Host Staging):**
```
源 GPU → cudaMemcpy → 主机内存 → RDMA NIC → 网络 → RDMA NIC → 主机内存 → cudaMemcpy → 目标 GPU
```

**GDR 零拷贝路径 (本实现):**
```
源 GPU 显存 → BAR1 窗口 → RDMA NIC → 网络 → RDMA NIC → BAR1 窗口 → 目标 GPU 显存
                ↑ CPU 与主机内存全程不参与 ↑
```

### SGLang 侧适配

新增 `sglang/srt/disaggregation/gdr/` 后端，作为 Mooncake/NIXL 的同级传输引擎：

1. **registration.py** — `GpuMrCache`: GPU 内存 LRU 注册缓存
   - 双路径自动选择: `peermem` (ibv_reg_mr on 设备VA) 或 `dma-buf` (CUDA 13+ ibv_reg_dmabuf_mr)
   - 每次注册前强制 `cuPointerSetAttribute(SYNC_MEMOPS=1)` 防止数据损坏
   - BAR1 预算超限时 LRU 淘汰 refcount=0 的旧 MR
   - `invalidate_stale()`: 检测 CUDA BUFFER_ID 变化, 失效过期条目

2. **transport.py** — `GdrTransport`: verbs QP/CQ 管理
   - 每个 (peer, device) 对一个 RC QP
   - TCP 侧信道交换 `{LID, GID, QPN, PSN, rkey, addr}` (仿 3FS IBConnectService)
   - QP 状态转换: RESET → INIT → RTR → RTS
   - `RDMA_WRITE_WITH_IMM` 批量提交, 接收端通过 IMM 获得完成通知
   - 后台 CQ 轮询线程 + per-room 完成队列

3. **conn.py** — `GdrKVManager/Sender/Receiver`: SGLang 抽象层实现
   - 继承 `BaseKVManager/Sender/Receiver` + `StagingManagerMixin`
   - Sender: 按 layer 批量调度 RDMA-WRITE, 使用 `group_concurrent_contiguous` 合并索引
   - Receiver: 广播目标 KV MR (rkey+addr), 通过 CQE 计数判断完成
   - 消费前调 `flush_gpudirect_writes()` 注入 NIC→CUDA happens-before 屏障

激活: `ServerArgs(disaggregation_transfer_engine="gdr", gdr_worker_threads=4)`

### 3FS 侧适配

改造 3FS 的 RDMA 传输层与 USRBIO Iov：

1. **GpuMemMgr.h/.cc** — GPU 内存管理子系统 (CUDA VMM API)
   - `GpuMemMgr::probe()`: dlopen libcuda, 检测 CUDA 可用性
   - `GpuMemMgr::isDevicePointer()`: 通过 CU_POINTER_ATTRIBUTE_MEMORY_TYPE 区分 CPU/GPU
   - `GpuMemMgr::setSyncMemops()`: 强制 SYNC_MEMOPS
   - `GpuMemMgr::exportDmabufFd()`: cuMemGetHandleForAddressRange
   - `GpuMemMgr::vmmAlloc()`: VMM 分配 (cuMemAddressReserve → cuMemCreate → cuMemMap → cuMemSetAccess)
   - `GpuMemMgr::vmmImportFromIpcFd()`: 跨进程导入 VMM 分配 (cuMemImportFromShareableHandle)
   - `GpuMrCache`: LRU pin 缓存, peermem/dmabuf 双路径
   - `GpuBufferPool`: 固定大小 GPU 缓冲池, VMM 单次分配多 slot

2. **RDMABuf.h/.cc.patch** — GPU 感知注册
   - `detectGpuPointer_()`: 自动检测指针是 GPU 还是主机
   - GPU 路径路由到 `GpuMemMgr::cacheFor(dev)->registerRegion()`
   - `sge()` 返回设备 VA (dma-buf MR 的 `mr->addr` 无意义)
   - `remoteAddrRkey()` 返回 (addr, rkey) 供 peer 构造 RDMA-WRITE

3. **Shm.cc.patch** — USRBIO Iov GPU 显存支持 (CUDA VMM API)
   - `mapBufGpu_()`: `GpuMemMgr::vmmAlloc()` (cuMemCreate/cuMemMap) 替代 `shm_open+mmap`
   - RAII: `shared_ptr<GpuVmmAllocation>` 管理生命周期, 析构逆序释放 (cuMemUnmap/cuMemRelease/cuMemAddressFree)
   - 跨进程 IPC: `cuMemExportToShareableHandle` 导出 POSIX fd, 用户进程通过 `vmmImportFromIpcFd()` 导入
   - 多 IB 设备预注册 (GpuMemMgr 缓存预热)
   - `HF3FS_IOV_USE_GPU_MEMORY=1` 环境变量开关
   - USRBIO 暴露 `gpuIovBase()` / `iovIpcFd()` 供应用直接获取设备指针

## 安装与使用

### 前置条件

- NVIDIA GPU (SM 80+: A100/H100/H200)
- Mellanox ConnectX-6 Dx 或更新 NIC (与 GPU 共享 PCIe root complex)
- NVIDIA 驱动 R470+
- `modprobe nvidia-peermem` (传统路径) 或 CUDA 13+ (dma-buf 路径)
- `pip install ibverbs` (Python RDMA 绑定)
- rdma-core >= 33 或 MLNX_OFED >= 5.x (含 `ibv_reg_dmabuf_mr`)

### SGLang 集成

```bash
# 1. 将模块复制到 SGLang 树
cp -r sglang_gdr/ <sglang-repo>/python/sglang/srt/disaggregation/gdr/

# 2. 应用集成补丁
cd <sglang-repo>
patch -p1 < /workspace/sglang_gdr/patches/parallel_state.py.patch
patch -p1 < /workspace/sglang_gdr/patches/utils.py.patch

# 3. 安装 Python 依赖
pip install ibverbs

# 4. 运行时配置
source /workspace/config/gdr_env.sh

# 5. 启动服务
python -m sglang.launch_server \
    --disaggregation-transfer-engine gdr \
    --gdr-worker-threads 4 ...
```

### 3FS 集成

```bash
# 基线: 3FS commit f84e609692dded708f9c910caf5c5af3444e1024
# cd <3fs-repo> && git checkout f84e609692dded708f9c910caf5c5af3444e1024

# 1. 复制源文件到 3FS 树
cp threefs_gdr/include/GpuMemMgr.h <3fs-repo>/src/common/net/ib/
cp threefs_gdr/src/GpuMemMgr.cc    <3fs-repo>/src/common/net/ib/

# 2. 应用补丁
cd <3fs-repo>
patch -p1 < /workspace/threefs_gdr/patches/IBDevice.h.patch
patch -p1 < /workspace/threefs_gdr/patches/RDMABuf.h.patch
patch -p1 < /workspace/threefs_gdr/patches/RDMABuf.cc.patch
patch -p1 < /workspace/threefs_gdr/patches/Shm.cc.patch

# 3. 将 CMake 片段合并到构建系统 (参考 config/CMakeLists.gdr.txt)
# 4. 编译
cmake -DHF3FS_ENABLE_GDR=ON .. && make -j

# 5. 运行时配置
export HF3FS_IOV_USE_GPU_MEMORY=1
export HF3FS_IOV_GPU_DEVICE_ID=0
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SGLANG_GDR_IB_DEVICE` | `mlx5_0` | IB 设备名 |
| `SGLANG_GDR_IB_PORT` | `1` | IB 端口号 |
| `SGLANG_GDR_GID_INDEX` | `3` | GID 索引 (3=RoCEv2) |
| `SGLANG_GDR_CQ_DEPTH` | `1024` | CQ 深度 |
| `SGLANG_GDR_MAX_WR_SEND` | `256` | 发送 WQ 深度 |
| `SGLANG_GDR_SIGNAL_EVERY` | `64` | 每 N 个 WR 产生一个 signalled CQE |
| `SGLANG_GDR_IMM_NOTIFY` | `1` | 使用 RDMA-WRITE-WITH-IMM |
| `SGLANG_GDR_PREFER_DMABUF` | `0` | 1=优先 dma-buf 路径 |
| `HF3FS_IOV_USE_GPU_MEMORY` | `1` | 3FS Iov 使用 GPU 显存 |
| `HF3FS_IOV_GPU_DEVICE_ID` | `0` | GPU 设备 ID |
| `HF3FS_GDR_BAR1_BUDGET_BYTES` | 8GiB | BAR1 预算上限 |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | dma-buf 导出必需 |

## 测试

```bash
# Python 单元测试 (无需 GPU/IB 硬件)
cd /workspace && python -m pytest tests/sglang/ -v

# 集成冒烟测试
bash tests/integration/run_smoke_test.sh

# C++ 单元测试 (需要 g++, 无需 CUDA/verbs 头文件)
g++ -std=c++17 tests/threefs/test_GpuMemMgr.cpp -o /tmp/test_gdr -pthread && /tmp/test_gdr
```

测试结果: **23 passed, 2 skipped** (Python), **5 passed, 0 failed** (集成)

## 关键技术点

1. **双注册路径**: `peermem` (ibv_reg_mr on 设备VA, 需 `modprobe nvidia-peermem`) 与 `dma-buf` (CUDA 13+ `cuMemGetHandleForAddressRange` + `ibv_reg_dmabuf_mr`, 无需内核模块), 运行时自动选择

2. **同步正确性**: 每次 pin 前强制 `cuPointerSetAttribute(SYNC_MEMOPS=1)`; 接收端消费前调 `flush_gpudirect_writes()` 注入 NIC→CUDA happens-before 屏障

3. **LRU 注册缓存**: 按 `(ptr, nbytes, BUFFER_ID)` 去重, BAR1 预算超限时淘汰 refcount=0 的旧 MR, 把毫秒级 pin 成本摊销到 ~0

4. **正确 advertise**: `GpuMrHandle.addr` 始终是设备 VA (dma-buf 的 `mr->addr` 无意义), peer `remote_addr` 据此构造

5. **双路径导入**: `__init__.py` 和 `conn.py` 使用 try/except 双路径导入, 既能安装到 SGLang 树内, 也能独立运行/测试

6. **CUDA VMM API (cuMemCreate/cuMemMap) 而非 cudaMalloc**: 所有 GPU 显存分配走 VMM 路径 (`cuMemAddressReserve → cuMemCreate → cuMemMap → cuMemSetAccess`), 不使用 `cudaMalloc`/`cuMemAlloc`, 因为:
   - VMM 是唯一可靠支持 `cuMemGetHandleForAddressRange` (dma-buf fd 导出) 的分配方式
   - VMM 提供显式 2 MiB 对齐, 匹配 GPU BAR1 页粒度, 最大化映射效率
   - VMM 分配可通过 `cuMemExportToShareableHandle` 导出 POSIX fd IPC handle, 供 FUSE 进程与用户应用进程共享同一物理分配
   - `GpuVmmAllocation` RAII 封装确保生命周期管理: 析构逆序释放 (`cuMemUnmap → cuMemRelease → cuMemAddressFree`)
