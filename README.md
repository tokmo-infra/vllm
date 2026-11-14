# Tokmo vLLM · Kimi3.1 HiSparse PD

Tokmo 的 Kimi3.1 Pro 推理分支，面向双机 B300 的 Prefill/Decode 分离部署。P 在 B300-92，HiSparse D 在 B300-93，两侧均为 TP8 / DCP8 / EP8。本分支保留 `latency_specialized`、DSpark4 和 FP32 recurrent 状态，重点优化长上下文下的 KV 容量与并发。

当前分支：`kimi31-dcp-hisparse-pd`。**当前启动参数统一放在 [tokmo/](tokmo/)**，P、D 使用独立脚本，每个服务只有一份 Docker 命令。这是 2026-11-14 对现有 92/93 部署的参数快照，镜像和命令已经与运行中容器核对。

## 分支改动

- HiSparse 支持 DCP 开启时的跨机 PD Decode，完整 source KV 保存在主存，HBM 保留 hot KV。
- DCP indexer logits 按本卡有效长度分配，覆盖普通路径、专用 fallback 和 profile；其他用途保留全局长度。
- KDA 按最终物理块布局重打包，69 层从 12 组压到 9 组，保留 FP32 recurrent，并统一调度、worker、PD 传输和 ReplaySSM 布局。
- D 侧 hot buffer 调整为 6144 行，支持 DSpark4 的 Q5 查询分块，保持完整 top-k 读取。
- DCP HiSparse D 容量抢占时隔离受影响请求，返回可重试的 HTTP 503，避免恢复成本地 Prefill 后退出整个引擎。
- Rust 前端复用 P 侧 prompt IDs，Router 透传历史缓存 usage；跨机 KV 传输使用 MooncakeConnector。

## 当前服务参数

| 参数 | P：B300-92 | D：B300-93 |
| --- | --- | --- |
| GPU / 并行 | 8 × B300，TP8 / DCP8 / EP8 | 8 × B300，TP8 / DCP8 / EP8 |
| 服务地址 | `11.5.0.92:19580` | `11.5.0.93:19581` |
| API / renderer workers | 1 / 1，Rust 前端 | 1 / 1，Rust 前端 |
| max model length | 1,048,576 | 1,048,576 |
| max num seqs | 256 | 256 |
| max batched tokens | 8192 | 2048 |
| GPU memory utilization | 0.97 | 0.95 |
| 实测 HBM KV 池 / 卡 | 46.328 GiB，1804 块 | 44.896 GiB，6993 块 |
| HiSparse hot / source | — | 6144 行 / 1024 GiB 逻辑主存池 |
| Prefix caching / Mamba mode | 开启 / `align` | 关闭 / `none` |
| 目标 KV / indexer KV | NVFP4 / MXFP4 | NVFP4 / MXFP4 |
| recurrent 状态 | FP32 | FP32 |
| 推测解码 | DSpark，4 draft tokens，FP8 draft KV | DSpark，4 draft tokens，FP8 draft KV |
| CUDA graphs | `FULL_AND_PIECEWISE`，尺寸见脚本 | `FULL_DECODE_ONLY`，21 图，最大 1280 |
| 专用 Decode backend | — | `latency_specialized` |
| 缓存连接器 | Mooncake producer + Store | HiSparse + Mooncake consumer |
| Store SSD 读取预算 | 128 MiB / 接收线程 / rank，4 接收线程 | — |
| Tokenizer cache | 12 GiB / API | 12 GiB / API |
| Watermark | 命令未显式覆盖 | 0.05 |

HBM KV 数值包含当前混合缓存布局的池内状态，不能直接当作每个请求都可独立使用的 token 容量。D 的 1024 GiB source 是 8 个 DCP ranks 合计的逻辑预算，每 rank 128 GiB；source 与共享 Store L2 分开计量。服务 `max-num-seqs=256` 与客户端目标并发独立，压测活跃用户配置始终为并发的两倍。

## 缓存层级

P 优先使用 HBM 和共享主存 L2，L2 淘汰后按需写入 SSD L3；D 使用自己的 HiSparse source 主存池和 HBM hot 池。P 不额外分配嵌入式 Store segment，`global_segment_size=0`；独立 owner 提供共享 L2。

| 节点 / owner | L2 / GiB | L3 配额 / GiB | SSD staging / GiB |
| --- | ---: | ---: | ---: |
| 92 / 0 | 496 | 1328 | 1 |
| 92 / 1 | 496 | 2272 | 1 |
| 92 / 2 | 560 | 2800 | 1 |
| 92 / 3 | 560 | 3184 | 1 |
| 93 / 0 | 320 | 1680 | 1 |
| 93 / 1 | 320 | 2304 | 1 |
| 93 / 2 | 304 | 3184 | 1 |
| 93 / 3 | 304 | 3184 | 1 |
| **总计** | **3360** | **19936** | **8** |

Master 在 `11.5.0.92:51561`，metadata 端口 19590，metrics 为 `127.0.0.1:20503`。策略为 `free_ratio_first`、`offload_on_evict=true`、`offload_force_evict=false`，命中两次可触发 promotion，lease 为 60 秒。L3 是按需增长的配额，表中配额不代表实时占用。

## 文件入口

| 文件 | 用途 |
| --- | --- |
| [tokmo/prefill-92.sh](tokmo/prefill-92.sh) | P 独立启动命令 |
| [tokmo/decode-93.sh](tokmo/decode-93.sh) | HiSparse D 独立启动命令 |
| [tokmo/router-92.sh](tokmo/router-92.sh) | PD Router，端口 19500，固定 P/D 地址 |
| [tokmo/mooncake-master-92.sh](tokmo/mooncake-master-92.sh) | Mooncake Master |
| [tokmo/store/](tokmo/store/) | 八个独立 owner 命令及 P 侧 Store 配置 |
| [tokmo/platform-profile.sh](tokmo/platform-profile.sh) | 设置 / 核验 CPU performance governor |
| [tokmo/bind-api-numa.sh](tokmo/bind-api-numa.sh) | 将 Rust API 线程绑定到 GPU0 所在 NUMA 的 CPU |
| [tokmo/deployment.json](tokmo/deployment.json) | 镜像 ID、运行容器身份、代码版本、预算和命令校验摘要 |

每个服务脚本默认 `--dry-run`，只打印完整命令；`--execute` 才会创建容器。脚本会检查目标节点和同名容器，保留已存在的实例。P/D 启动不经过共同 launcher。

## 在现有 92/93 环境启动

本快照使用节点本地的定制镜像 ID，统一 `--pull=never`。完整 ID 见脚本和 `deployment.json`，这些 ID 不是公开镜像下载地址。启动前需要准备好模型、草稿、运行目录和脚本使用的 RDMA 网卡 / GID3：

- 目标模型：两机的 `/data/models/Kimi-K3.1-Pro`，挂载为 `/model`。
- 草稿模型：92 使用 `/data2/bench/eng-yangyufei/kimi31-slo-20261110/draft`；93 使用 `/data2/bench/eng-yangyufei/kimi31-slo-b30093-20261110/draft`，均挂载为 `/draft`。
- 运行目录：两机的 `/data2/bench/eng-xinhao/hisparse-pd-correctness-20261113-ced31f9/full`，挂载为 `/work`。
- 保留现有运行目录中的 `pd_admin.py`、`pd_cache_probe.py`、`runtime/` 和缓存目录。启动参数引用 `pd_admin.CapacityObserver`；这些验收扩展和模型 / 镜像不包含在本参数快照中。
- P 的 `/work/store-plan/mooncake-92.json` 应使用 [tokmo/store/mooncake-92.json](tokmo/store/mooncake-92.json)。D 不配置 Store client JSON。
- Owner 使用脚本指定的 SSD 目录、NUMA 和 CPU 列表；新建部署前核对对应挂载和可用空间，保留现有缓存数据。

在仓库根目录预览两侧命令：

```bash
bash tokmo/prefill-92.sh
bash tokmo/decode-93.sh
```

下面是空闲目标节点的启动顺序。已运行的压测继续使用原实例，不执行重启：

1. **92：** Master → 四个 92 owner → P。
2. **93：** D 初始化就绪、实际分配 source 后 → 四个 93 owner。
3. **92：** P/D 和八个 owner 就绪后启动 Router。

92 上执行：

```bash
bash tokmo/mooncake-master-92.sh --execute
for i in 0 1 2 3; do
  bash "tokmo/store/mooncake-owner-92-$i.sh" --execute
done
bash tokmo/platform-profile.sh --apply
bash tokmo/prefill-92.sh --execute
bash tokmo/bind-api-numa.sh kimi31-pd-hisparse-92-prefill-c256-r17-gpu097 http://11.5.0.92:19580/health
```

93 上执行；NUMA 绑定脚本会等待 D 的 health 就绪，再启动 owner：

```bash
bash tokmo/platform-profile.sh --apply
bash tokmo/decode-93.sh --execute
bash tokmo/bind-api-numa.sh kimi31-pd-hisparse-93-decode-c256-r17-gpu095 http://11.5.0.93:19581/health
for i in 0 1 2 3; do
  bash "tokmo/store/mooncake-owner-93-$i.sh" --execute
done
```

P/D 和 Store 就绪后，在 92 启动 Router：

```bash
bash tokmo/router-92.sh --execute
```

对外 API 为 `http://11.5.0.92:19500`，模型名 `kimi-k3.1-pro`。CPU governor 和 NUMA 绑定需要节点已有的免交互 sudo 权限；GPU 时钟保持现有部署设置。

## 验证范围与性能口径

当前实现已完成真实跨机 PD 验证：13 个长度边界至 1,048,531 tokens、240k 历史检索、256/256 并发校验码、8 ranks 共 90,150 行 source/hot 字节核对。强制容量抢占验证中，只有被抢占请求返回 503，D 未重启，后续已知答案请求通过。Router 冷 / HBM 命中 / Store 命中的 cached usage 为 0 / 32768 / 32768；P 的约 130k / 1M 输入真实 SSD 恢复零 load 失败。

实现提交：`bab602a8ab34dd29bb35ec1163e7f500d6dffa8e`（容量优化），`5caa5d7f5c989783ba90ba92c57e521660e8fb9d`（容量抢占隔离）。现有 P 镜像对应前者，D 镜像对应后者。验证通过的路径不等同于完整跨后端数值等价或 SWE 全量准确率评估；生产 L3 满配额淘汰尚未全面验证。

性能目标为 **Mean TTFT ≤ 5 秒、Mean TPOT ≤ 50 毫秒**，活跃用户 A=2C。以本轮 40% 进度后的连续完整、并发达标分钟统计，平均输入约 130k 或自然欠载时冻结；分母为 P8+D8 共 16 个物理 ranks。更改 A 会改变有限语料 cohort，不能只用最高并发认定最高合格 TPM。已完成的 C256/A512、平均输入 130,112 的正式窗口为 27.485M TPM，但 Mean TTFT 24.188 秒，未满足 SLO；最高合格 TPM 仍在调优。

本次参数归档通过 12 个运行容器的 220 项只读 Docker 检查，12 份脚本 dry-run 的 argv 与部署命令逐项一致。参数导出本身只读。当前 P0.97/D0.95 已重新 profile 和完成 CUDA Graph 捕获，131073 / 1048531 token 的真实 PD 已知答案及 Router 六项检查通过。D0.97 在 DSpark logits all-gather 额外分配300MiB时 OOM，按用户选择保持 D0.95；FP32、hot6144 和专用后端保持不变。

## 上游与许可

本仓库基于 [vLLM](https://github.com/vllm-project/vllm)，保留上游源码和版权声明。采用 [Apache-2.0](LICENSE) 许可；上游通用文档保留在 [docs/](docs/)。

## 开发与反馈

<!-- --8<-- [start:contact-us] -->
本分支的部署问题与改动记录请提交到 [Tokmo vLLM Issues](https://github.com/tokmo-infra/vllm/issues)，并附启动参数、代码版本和复现日志。
<!-- --8<-- [end:contact-us] -->
