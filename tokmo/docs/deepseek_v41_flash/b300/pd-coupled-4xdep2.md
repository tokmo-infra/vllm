# Tokmo DeepSeek-V4.1-Flash B300：4×DEP2 P/D 耦合

本文维护企业自定义镜像的生产推荐配置：**单机 8×B300，4 个 TP1/DP2/EP2 的 P/D 耦合实例，1600GiB 共享 DRAM L2，2TiB SSD L3**。模型为 DeepSeek-V4.1-Flash-1017。脚本按模型、硬件和部署拓扑存放，不依赖 Git 分支名称。

配置入口：[部署脚本目录](../../../deployments/deepseek_v41_flash/b300/pd-coupled-4xdep2/README.md)。本文仅保留这一套容量和拓扑。该完整容量组合已完成静态参数检查，尚未完成新的服务压测；生产推荐配置不等同于该组合已经验收上线。

## 镜像与代码来源

| 项目 | 值 |
| --- | --- |
| 服务镜像 | `local/vllm-openai:0.32.0-dsv41-native-token-cache-v1` |
| B300 原镜像 ID | `sha256:e0fdb06bdeecb93075241a839e2b943b26019a9b4bf0c1c2df0b7ba4f0717c99` |
| 基础镜像 | `vllm/vllm-openai:v0.32.0`，Linux/amd64 |
| 基础镜像 ID | `sha256:bd8a5d7a8774800c92d714c0c226b32c1863f929150f36efd610aebcd12a335d` |
| vLLM 基础源码 | `5df63a7203cdb9328d824a27b682e77c7b9928be`，`v0.32.0` |
| Router 镜像 | `local/vllm-router:0.1.16-xinhao` |
| Router 镜像 ID | `sha256:b00a9c082fa6038662a514d829041a7f212fca002ee1c3754c9dbbb7acdfe498` |
| Mooncake SDK | `mooncake-transfer-engine-cuda13==0.3.13.post1` |
| Torch / Transformers / Tokenizers | `2.13.0+cu130` / `5.19.0` / `0.23.3` |

2026-10-30 从 B300 原镜像提取并比对源码：3293 个上游已跟踪包文件完全相同，修改 `vllm/renderers/base.py`，新增 `vllm/tokenizers/dsv41_native_cache.py` 和 `vllm/tokenizers/dsv41_segment_cache_core.py`。这三个文件在仓库中保留镜像原字节；其余基础源码保留完整上游历史。镜像的 examples/benchmarks 与对应上游文件一致。

自定义部分在原生 renderer 内增加 CPU 精确分段分词缓存和前端计时；沿用原生分词结果、cache salt 与回退路径。它与 GPU KV、Mooncake L2/L3 是不同缓存。GPU kernel、模型计算和原生二进制沿用基础镜像。第三方构建依赖、生成文件及二进制的安装态清单见 [installed-vllm.sha256](../../../images/deepseek_v41_flash_b300/installed-vllm.sha256)，镜像依赖见 [image-packages.txt](../../../images/deepseek_v41_flash_b300/image-packages.txt)，来源信息见 [image-info.json](../../../images/deepseek_v41_flash_b300/image-info.json)。

原镜像和基础镜像在 B300 上没有 `RepoDigests`。这里记录的 SHA256 是 Docker **image ID**，不可当作仓库 manifest digest 使用。将镜像推入企业 Registry 后，应另外固定该 Registry 返回的 digest。仓库中的构建配方生成代码一致的新镜像，不承诺重建出相同 image ID。

## 镜像构建

已有原镜像时可直接使用。需要从仓库重建时，在 B300 的仓库根目录执行：

```bash
bash tokmo/deployments/deepseek_v41_flash/b300/pd-coupled-4xdep2/build-image.sh
```

[构建脚本](../../../deployments/deepseek_v41_flash/b300/pd-coupled-4xdep2/build-image.sh)检查本地基础镜像 ID，以 `tokmo/images/deepseek_v41_flash_b300/Dockerfile` 构建，复制仓库内三个实际源码文件，并检查 SHA256 与 Python 编译。默认禁止拉取基础镜像和构建时访问网络。基础镜像、CUDA 扩展和依赖由已验证的底座提供。

`IMAGE` 可指定输出 tag，`BASE_IMAGE` 可指定同一基础镜像的本地名称；两者都必须已具备相应输入。只有明确升级并重新验收底座时才修改 `EXPECTED_BASE_IMAGE_ID`。Router 是独立部署依赖，不由此 Dockerfile 构建。

## 计算拓扑与推理参数

| 实例 | GPU | NUMA | HTTP | DP RPC |
| --- | --- | --- | ---: | ---: |
| 0 | 0、1 | 0 | 30020 | 16360 |
| 1 | 2、3 | 1 | 30022 | 16460 |
| 2 | 4、5 | 2 | 30024 | 16560 |
| 3 | 6、7 | 3 | 30026 | 16660 |

[coupled.sh](../../../deployments/deepseek_v41_flash/b300/pd-coupled-4xdep2/coupled.sh)保留以下设置：

- 每实例 TP1/DP2/EP2，EP 权重过滤、EPLB、NUMA 绑定；每 rank `max-num-seqs=512`、`max-num-batched-tokens=16384`，模型上下文由 `max-model-len=-1` 自动选择。
- `FP8` KV、`FLASHMLA_MEGA_ATTN_DSV41` attention、`mxfp4` indexer、`deep_gemm_mega_moe`、`deep_select`；开启 GPU prefix cache。
- DSpark 固定 5 个投机 token，关闭 adaptive verification。
- Engram 使用 `cpu_offload=true`、`embedding_across_dp=true`、`dp_shared_memory=true`，每实例约 188.83GiB，全机约 755.32GiB。
- 每实例 8 个 API 进程，每进程 1 个 renderer worker、4GiB CPU 分词缓存；全机缓存上限 128GiB。每实例有独立 tokenizer epoch 和编译缓存目录。
- 原生 `deepseek_v41` tokenizer、reasoning/tool parser，自动工具选择；`enable-prompt-tokens-details` 暴露缓存统计。

四个实例共用同一个 Mooncake master 和稳定 namespace：`dsv41-flash-1017-b300-pd-coupled-4xdep2-l2-200g-l3-2t-v1`。模型、精度、KV 布局或 tokenizer 身份改变时，应分配新的 namespace 与持久目录。

## KV cache 卸载与 L2/L3

| 项目 | 每 owner / rank | 全机 |
| --- | ---: | ---: |
| 共享 DRAM L2 | 200GiB/owner | 1600GiB |
| SSD L3 配额 | 256GiB/owner | 2TiB，即 2048GiB |
| SSD 临时 staging | 1.25GiB/owner | 10GiB，不计入 L2 |
| vLLM 私有 local buffer | 4GiB/rank | 32GiB，不计入 L2 |
| owner 容器内存上限 | 224GiB | 8 个 owner |
| master 容器内存上限 | — | 64GiB |

[mooncake.json](../../../deployments/deepseek_v41_flash/b300/pd-coupled-4xdep2/mooncake.json)设置 `mode=standalone-store`、`global_segment_size=0`。vLLM rank 只作为请求方，200GiB L2 由每个独立 owner 持有，不能再次给 8 个 vLLM rank 分配共享 segment。所有实例都可使用整个共享池，未限制到固定 owner 或 preferred segment。

8 个 owner 在 [mooncake-owner.sh](../../../deployments/deepseek_v41_flash/b300/pd-coupled-4xdep2/mooncake-owner.sh)中分别设置 `global_segment_size=214748364800`，L3 的逻辑、物理和容量声明上限均为 `274877906944` bytes。每个 owner 使用自己的 `owner-0` 至 `owner-7` 子目录；默认父目录为：

```text
/data1/lab/eng-xinhao/mooncake-l3/b300-pd-coupled-4xdep2-l2-200g-l3-2t-v1
```

`MOONCAKE_L3_ROOT` 可覆盖父目录，8 个 owner 应使用相同父路径。目录沿用现有 `/data1` NVMe 文件系统，不预分配填满磁盘。部署前确认它可容纳 2TiB 配额并留有其他业务余量。单个 owner 的配额不会自动借给另一个 owner。

| 策略 | 配置 |
| --- | --- |
| master 卸载 | `enable_offload=true`、`offload_on_evict=true`、`offload_force_evict=false` |
| L2 晋升 | `promotion_on_hit=true`、`promotion_admission_threshold=2` |
| SSD 后端 / 淘汰 | `bucket_storage_backend` / `lru` |
| 后台水位淘汰 | 启用，高 90%、低 80% |
| I/O | `MOONCAKE_OFFLOAD_USE_URING=true`，物理占用扫描缓存 500ms |
| 推理端 | `MooncakeStoreConnector`、`kv_both`、`lookup_async=true`、`load_async=true` |
| master / owner 重启 | `unless-stopped` |

写入 L2 后，内存淘汰选中的 KV 下沉 L3；SSD 同样执行有界淘汰。读取路径为 **NVMe → owner 临时 DRAM staging → TransferEngine/RDMA → 请求方 GPU KV buffer**，不要求先晋升回共享 L2。此配置未启用 GDS。B300 上固定缓冲注册曾返回 EFAULT 后回退普通 io_uring，部署时检查实际日志，不能仅根据开关宣称使用了固定缓冲。

bucket 后端可在 owner 重启时扫描持久元数据并向 master 重新登记 SSD 副本。普通启停保留目录；重启后 DRAM 冷启动，需要重新等待健康和容量登记。该部署使用单 master、单节点、单 NVMe，没有跨节点高可用或磁盘冗余。

## HCA、NUMA 和容器设置

此配置对应 B300 主机 `11.5.0.55`。GPU0–7 之间为 NVLink；GPU 对依次属于 NUMA0–3。owner `i` 绑定 NUMA `i/2` 的 CPU 列表与内存节点，模型进程使用与 GPU 对对应的 `--numa-bind-nodes`。

Mooncake 请求端 `device_name=mlx5_6`，所有 owner 使用 `--device_names=mlx5_6`，协议为 `rdma`，metadata 为 `P2PHANDSHAKE`。2026-10-30 只读检查中，该 HCA 位于 NUMA1，port 1 为 ACTIVE、400Gb/s、Ethernet/RoCE；`nvidia_peermem` 已加载。这是当前配置选择，并不表示所有 GPU/owner 均与这张 HCA 在同一 NUMA 节点，也不代表已经测得 400Gb/s 有效带宽。

迁移机器时，先检查 `nvidia-smi topo -m`、`/sys/class/infiniband/` 下端口 state/rate/link_layer 与设备 numa_node，再同时修改请求端 JSON、owner 的 HCA 和服务 IP。当前没有手动指定 GID、TC/SL、`NCCL_IB_HCA`、`NCCL_SOCKET_IFNAME`、`UCX_NET_DEVICES`；不要将其他拓扑的值直接套入本配置。

模型容器使用 host network、host IPC、`/dev/infiniband`、`IPC_LOCK`/`SYS_NICE`、无限 memlock、64MiB stack 和 `seccomp=unconfined`。owner 使用 NVIDIA runtime 且 `NVIDIA_VISIBLE_DEVICES=none`，用于加载驱动库；master/Router 不占用 GPU。模型容器向 vLLM 暴露全部 GPU，具体实例通过 `--device-ids` 选择对应 GPU 对。

静态内存预算：L2 1600GiB + Engram 755.32GiB + 私有 buffer 32GiB + tokenizer 上限 128GiB + staging 10GiB，合计约 **2525.32GiB**，另加 master/owner 元数据、Router 和其他进程开销。容器内存上限不等于预分配量，启动前应检查整机和各 NUMA 节点的可用内存。

## 启动顺序

依赖：NVIDIA 驱动及容器 runtime、RDMA 设备、已验证的三个本地镜像（基础/自定义/Router）、模型权重、Python 3.12 与 `uv`。默认权重路径 `/data2/models/DeepSeek-V4.1-Flash-1017`，可用 `MODEL_DIR` 覆盖。模型容器仅离线读取权重。

在 B300 仓库根目录执行：

```bash
cd tokmo/deployments/deepseek_v41_flash/b300/pd-coupled-4xdep2
uv venv --python 3.12 .venv

bash mooncake-master.sh
for owner in {0..7}; do bash mooncake-owner.sh "$owner"; done
.venv/bin/python check_mooncake.py

for instance in {0..3}; do bash coupled.sh "$instance"; done
# 等待四个模型 /health 全部返回 200，CPU Engram 初始化可能约需 30 分钟。
for port in 30020 30022 30024 30026; do
    until curl --noproxy '*' -fsS "http://127.0.0.1:$port/health"; do sleep 10; done
done
bash router.sh
curl --noproxy '*' -f http://127.0.0.1:18086/health
curl --noproxy '*' -f http://127.0.0.1:18086/workers
```

`check_mooncake.py`只读检查 8 个健康 owner、每份 200GiB L2、总计 2TiB L3 登记和对应 segment，不要求缓存为空，也不发送生成请求。脚本遇到同名容器会拒绝创建；更新前先在维护窗口排空入口，核对容器身份后再执行替换。启动流程不会清理 SSD 或自动压测。

默认状态保存在脚本目录内的 `cache/`、`native-tokenizer-state/`、`validation/`，均被 Git 忽略。`WORKDIR` 可统一改到另一个持久状态目录；模型与配置文件仍以脚本目录为准。L3 目录由 `MOONCAKE_L3_ROOT` 单独控制。

## 入口、监控与维护

| 组件 | 端口 |
| --- | --- |
| Router API / metrics | 18086 / 18087 |
| 模型 HTTP | 30020、30022、30024、30026 |
| DP RPC | 16360、16460、16560、16660 |
| master RPC / metrics | 50066 / 9017 |
| owner RPC | 50080–50087 |
| owner TransferEngine | 50140–50147 |
| owner HTTP health / metrics | 9320–9327 |

[router.sh](../../../deployments/deepseek_v41_flash/b300/pd-coupled-4xdep2/router.sh)使用普通耦合路由，将四个后端展开为 8 个 DP worker，策略 `cache_aware`，内存上限 64GiB，请求超时 600 秒，worker 启动超时 7200 秒。API 地址 `http://11.5.0.55:18086/v1`，配置中的 key 为 `EMPTY`；面向企业用户的访问控制由入口层提供。

在独立终端执行 `.venv/bin/python monitor.py`，每 10 秒采集模型、Router、master、owner 与 NVMe 计数器，写入状态目录的 `validation/production-external-metrics.jsonl.gz`，Ctrl+C 结束。关注外部 KV 命中、SSD bytes/ops/latency、淘汰、promotion、TTFT/TPOT。外部 KV token 包含 L2 和 L3，不能全部当作 SSD 命中。

维护时保留 SSD 目录和稳定 namespace。停机先排空入口并停止 Router，再停止四个模型、8 个 owner，最后停止 master；升级后按相反顺序恢复并检查容量和健康。更换 tokenizer 或 KV 布局使用新 namespace；日常重启无需删除持久数据。
