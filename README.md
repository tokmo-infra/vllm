# GLM5.3 Flash 1103

本分支用于 GLM-5.3 Flash 的 Prefill / Decode 分离部署，提供原生推理、同机共享分词、Mooncake 缓存恢复和 RDMA 网卡亲和配置。已跑通的 4P4D 拓扑、镜像要求和完整启动命令统一见 [deployment4p4d.md](deployment4p4d.md)。

## 1. b1872978 · 原生推理与共享分词

| 功能 | 解决的问题 | 实现和作用 | 通用复用范围 / 配置 |
| --- | --- | --- | --- |
| ① GLM推理与KV传输 | 混合Mamba / MLA缓存需要正确分组、传输和恢复状态。 | 对接原生MTP3、FP8 KV、CUDA graph及异步调度，按完整逻辑块保存和交接。 | GLM推理适配；缓存分组与传输思路可复用。 |
| ② 同机共享精确分词 | 多API会重复分词、重复保存同一段文本。 | 同机共享分段结果，每个API保留64MiB私有L1；用Tokenizer指纹、命名空间和epoch隔离。 | 共享池通用；分段规则适配固定GLM Tokenizer，路径和容量见部署配置。 |
| ③ GET租约刷新 | 读取排队期间副本租约可能到期。 | 对返回-707的key重新查询副本并完整读取一次，其他结果交给原生失败处理。 | Mooncake读取保护，随connector路径生效。 |
| ④ 相邻checkpoint回退 | 最近的Mamba checkpoint可能缺失或group尚未齐全。 | 按完整group与TP命名空间匹配相邻可用边界，并补齐采样所需checkpoint。 | 混合缓存复用；沿用Store查询和保存路径。 |
| ⑤ FlashInfer自动调优 | 不同rank重复搜索会增加启动开销。 | 复用匹配的调优配置并保留原生预热流程。 | 兼容的硬件、库版本与形状之间复用。 |

## 2. b68a1d20 · 缓存保存恢复与大块空间回收

| 功能 | 解决的问题 | 实现和作用 | 通用复用范围 / 配置 |
| --- | --- | --- | --- |
| ① PUT压力恢复 | CPU缓存曾经分配失败后，发送线程可能一直跳过后续保存。 | 每发送线程最多每秒用一个当前有效job探测，成功后恢复保存。 | Mooncake Store通用恢复逻辑，随代码生效。 |
| ② Master实际占用计账 | 分配器向上取整，导致有效字节看似未满、实际空间已不足。 | 将bin实际占用计入淘汰水位，保留payload指标和快照含义。 | offset allocator通用修复，使用配套Master。 |
| ③ Master大块空间回收 | CPU缓存有空闲，但碎片放不下新的大块KV。 | 定期检查目标大块可分配槽数，提前触发原生淘汰，为保存腾出空间。 | 大对象缓存通用；MC_EVICTION_FIT_REQUEST_BYTES默认0，配套命令给出目标大小。 |

## 3. 3619d13a · 可选末段准入预算

| 功能 | 解决的问题 | 实现和作用 | 默认值 / 配置 |
| --- | --- | --- | --- |
| ① CoW状态块预算 | producer异步加载后，prefill末段需要额外状态块，竞争请求可能先用光余量。 | 为已识别的同末物理块请求登记义务，并在准入时保护余量，减少抢占重算。 | VLLM_EXPERIMENTAL_HOT_TAIL_ADMISSION默认0；P端精确设1开启。 |
| ② 有限事件观测 | 需要确认预算何时登记、延后请求和被消耗。 | 每个scheduler记录三类首事件，便于检查实际路径。 | VLLM_HOT_TAIL_ADMISSION_DIAGNOSTICS默认0，设1记录。 |

该预算适用于首次、纯外部异步加载、同一物理末块的GLM/Mamba producer请求。启用时保留原生异步、块引用、Store交接和GPU栅栏流程；按实际业务对照吞吐与延迟后选择使用。

## 4. 5aeb9bd9 · 目的网卡亲和与部署

| 功能 | 解决的问题 | 实现和作用 | 通用复用范围 / 配置 |
| --- | --- | --- | --- |
| 目的 HCA 亲和 | 跨机交接 KV 时，需要让 SDK 配合 GPU 与 RDMA 网卡的实际拓扑选择目的设备。 | 设置 MC_ENABLE_DEST_DEVICE_AFFINITY=1，结合每个 GPU 分组的本地 HCA 绑定使用。 | 支持该提示的 Mooncake RDMA 路径；网卡映射按硬件配置。 |

Router 的 prompt ID 复用让 D 侧复用 P 侧已有的分词结果，减少重复处理长 prompt。它要求 Router、P、D 使用相同模型的 tokenizer 和 chat template。复用实现移植自 `08d7195`，默认开启；部署仍显式设置 `VLLM_ROUTER_PD_REUSE_PROMPT_IDS=1`。

## 5. 已跑通的 4P4D 部署

这里的 4P4D 按主机计数：81–84 每台两个 DEP/EP4 Prefill 实例，共 **8P / 32 rank**；85–88 每台一个 DEP/EP8 Decode 实例，共 **4D / 32 rank**，合计 **64 rank**，TP 均为 1。

| 角色 | 参数 |
| --- | --- |
| P | 每 DP seq24 / batch2048；native auto graph 上限 192；4 API/实例；KV recv4。 |
| D | 每 DP seq32 / batch1024；FULL_DECODE_ONLY graph 上限 128；1 API/实例。 |
| 公共 | 原生 async、FP8 KV、MTP3、fullISL、水位 0；目的 HCA 亲和设 1。 |
| 缓存 | 每台 Store 750 GiB；Tokenizer 总预算 2 GiB/GPU，即 16 GiB/台，**已含**每 API 64 MiB 私有 L1。 |
| Router | 16 CPU / 16 GiB，同模型 tokenizer/template，登记全部 8P + 4D。 |

P 抢占优化的**引擎代码默认值仍为 0**，本 4P4D 部署命令在 P 端显式设置 `VLLM_EXPERIMENTAL_HOT_TAIL_ADMISSION=1`；诊断开关保持 0。这与 Router 复用默认 ON 是两个独立配置。

按 [deployment4p4d.md](deployment4p4d.md) 准备镜像、IP、路径和凭据，再依次启动 Master、Store、模型实例与 Router。该文档是本分支的主部署入口。
