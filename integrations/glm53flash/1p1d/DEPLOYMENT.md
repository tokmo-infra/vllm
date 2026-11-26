# GLM-5.3-Flash NVFP4 · 85/86 1P1D 部署归档

85 是 Prefill 节点，运行两套 DEP4；86 是 Decode 节点，运行一套 DEP8。客户端通过 85 的 Router 访问 `http://11.1.0.85:28300/v1`，模型名为 `glm-5.3-flash`。部署参数见 `deploy.env`。

## 节点与端口

| 节点 / 角色 | 实例 | GPU | RDMA HCA | HTTP / KV bootstrap / DP RPC |
| --- | --- | --- | --- | --- |
| 85 · `p` · `11.1.0.85` | `p0` · DEP4 | 0–3 | `mlx5_2,mlx5_3` | `28483 / 29083 / 29550` |
| 85 · `p` · `11.1.0.85` | `p1` · DEP4 | 4–7 | `mlx5_4,mlx5_5` | `28493 / 29093 / 29650` |
| 86 · `d1` · `11.1.0.86` | `d0` · DEP8 | 0–7 | `mlx5_2,mlx5_3,mlx5_4,mlx5_5` | `28483 / 29083 / 29550` |

两台业务网卡均为 `ens86np0`，RDMA GID index 为 `3`。Mooncake Master 在 86，RPC 端口为 `61741`，元数据和指标分别为 `61742/metadata`、`61743/metrics`。85、86 各有一个本服务的 Store，使用各自的业务 IP，RPC 端口为 `61744`，健康检查为 `61745/health`。Router 在 85，API 和指标端口为 `28300`、`28301`，P 使用 cache_aware，D 使用 round_robin（8 个 DP），请求超时为 3600 秒。Mooncake L2 淘汰高水位为 95%，L3 采用 LRU。

三套引擎均使用 TP1、专家并行、NVFP4 KV 和 256 token 前缀匹配单元。DFlash2 每步最多提议 5 token，草稿 KV 为 FP8。P 单实例的序列上限和批 token 上限分别为 24、2048；D 为 32、1024，使用 `FULL_DECODE_ONLY` CUDA Graph。捕获上限为 seq×6：P 144、D 192。

## 公共镜像构建与交付

在 Linux x86_64 的 6000D 构建节点，从源码包根目录执行：

```bash
mkdir -p /var/cache/glm53-c32-1126/build
BUILD_PARENT=/var/cache/glm53-c32-1126/build BUILD_JOBS=8 bash integrations/glm53flash/1p2d/build/build-all.sh source
TMPDIR=/var/cache/glm53-c32-1126/build bash integrations/glm53flash/1p1d/build-dflash.sh
```

第一步从公开的 PyTorch CUDA 13 构建镜像获取 SM120 工具链，并从固定源码构建 vLLM、FlashInfer 0.7.1、Mooncake 和 Router。第二步从 Docker Hub 的 Rust 1.95 镜像按 Cargo.lock 编译 Rust 前端，叠加 Rust/DFlash 的运行文件到同 ABI 的 NVFP4 镜像。

- **Rust**：`build-dflash.sh` 生成 `/usr/local/bin/vllm-rs`。启动时配置 `VLLM_USE_RUST_FRONTEND=1`、`VLLM_RUST_FRONTEND_PATH=/usr/local/bin/vllm-rs`，每实例一个 API 进程。
- **镜像**：两台导入 `local/glm53flash-dflash2:r15b-rust-pd-85-86`、`local/glm53flash-mooncake:source-71973589` 和 `router:glm53flash-api-v7-20261124`。主模型和 DFlash2 权重放在配置里的路径。
- **TileLang**：0.1.12，在引擎运行时按需编译。启动预热或首次执行相关算子时，缓存未命中便触发编译；重启后复用同一实例目录里仍匹配的缓存。冷启动留足编译时间，正式计时前用代表性请求覆盖相关路径。
- **系统盘编译缓存**：宿主机根目录为 `/var/cache/glm53-c32-1126`，85 使用 `p0`、`p1` 子目录，86 使用 `d0` 子目录。各实例的目录挂到容器的 `/root/.cache`，其中 `tilelang` 子目录另挂到 `/root/.tilelang`。部署前确认 `/var/cache` 所在系统盘的可用空间，重启和组间清业务缓存时保留这些目录。

已有上述同 ABI 的 NVFP4 镜像时，直接运行第二条命令即可构建 Rust/DFlash 层；可用 `BASE_IMAGE` 指定镜像。85/86 现场使用已验证基底派生的镜像；公开基底从零重建仍待实机走完。

## 部署顺序

两台的服务目录均为 `/data/services/glm53flash-1p1d-8586`。使用 root 执行，准备同内容的 `engine.key`（常规文件，`root:root`，权限 `0600`），并确认 GPU 空闲、网卡地址和 HCA 与配置匹配、所需端口可用、宿主机有 `libcuda.so.1`。将 `launch.sh`、`deploy.env` 和 `numa_wrapper.sh` 复制到两台服务目录；跑分节点另复制 `cache8586.py`。主模型 `/data/models/GLM-5.3-Flash-NVFP4` 和草稿模型 `/data/models/GLM-5.3-Flash-DFlash2` 均以只读方式挂入引擎。

Mooncake L2 为 85 上 700 GiB、86 上 650 GiB。每台 L3 上限为 2.5 TB（十进制），目录为 `/data/l3/glm53flash-1p1d-8586`；脚本要求 `/data` 独立挂载，且 L3 目录已有占用加所在盘可用空间至少为 2.5 TB 加 200 GiB。P→D KV 走 Mooncake RDMA；P 通过 Store connector 读写业务缓存，D 读取，命名空间为 `glm53flash-1p1d-8586-nvfp4-prefix256`。

启用组间清缓存时，在两个服务目录准备内容一致的 `cache-reset.key`：在一台用 `openssl rand -hex 32` 生成，安全复制到另一台，两份文件均由 root 持有、权限 `0600`。启动脚本用只读挂载开启鉴权清缓存入口。

按顺序在对应节点的服务目录执行：

1. **86 启动 Master**：`NODE_ROLE=d1 bash launch.sh master`。确认 `http://11.1.0.86:61743/metrics` 可访问后启动引擎。
2. **两台启动引擎**：85 执行 `NODE_ROLE=p bash launch.sh models`，86 执行 `NODE_ROLE=d1 bash launch.sh models`。可在任一节点设置本机的 `NODE_ROLE` 后运行 `bash launch.sh wait`，检查三个引擎的 `/health`。
3. **两台启动 Store**：85 执行 `NODE_ROLE=p bash launch.sh store`，86 执行 `NODE_ROLE=d1 bash launch.sh store`。脚本等引擎就绪后启动 Store，并检查本机 Store 的 `/health`。
4. **85 启动 Router**：`NODE_ROLE=p bash launch.sh router`。Router 启动后，向 `/v1/chat/completions` 发一条短请求。

## 跑分前验收

1. 按配置的 HCA 和 GID 验证 85 → 86 的 RDMA 连通；结合实际请求和引擎日志，确认 P0、P1 均可将 KV 交给 D0。
2. 检查 Master 指标、两台 Store 健康状态和缓存读写，用重复前缀验证缓存命中。检查 Router 流式响应和指标，确认两个 Prefill 后端及 Decode 后端均正常接入。
3. 每组跑分前清理并核验本服务的业务缓存，覆盖专用引擎前缀缓存以及本服务的 Mooncake L2/L3。此部署使用专用 Mooncake Master，清缓存会清空该 Master；排空请求后在 86 以 root 运行 `python3 cache8586.py clear`。保留系统盘编译缓存；预热请求与计时样本使用不同前缀。
4. 按现行 **C160、240 用户 / 240 样本** 口径执行正式跑分，取完成 55% 轮次后到首次持续欠载前的完整分钟。同窗 TTFT P50 严格小于 5 秒时并发加 10，用户和样本各取并发的 1.5 倍。
