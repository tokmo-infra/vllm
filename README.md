# GLM5.3 Flash 1103

本分支用于 GLM-5.3 Flash NVFP4 的 Prefill / Decode 分离部署。镜像构建、拓扑与启动命令见[1P2D部署](deployment1p2d.md)与[4P4D部署](deployment4p4d.md)，两者开头均有字段说明和使用示例。

## 推理与缓存

| 功能 | 配置与作用 |
| --- | --- |
| 原生推理 | 原生异步、CUDA graph、MTP3、FP8 KV、fullISL；注意力和大 batch MoE 保持默认后端。 |
| 同机共享精确分词 | 整机共享分段结果，每个 API 保留 64 MiB 私有 L1；Tokenizer 指纹、命名空间和 root-owned epoch 隔离。总预算 2 GiB/GPU，已含 L1。 |
| Mooncake PUT 恢复 | 缓存分配失败后，每发送线程最多每秒用一个有效 job 探测，成功后恢复保存。 |
| Mooncake GET 恢复 | 对 -707 key 刷新副本租约并完整重读；按完整 group 和相邻 checkpoint 恢复混合 Mamba / MLA 状态。 |
| Master 空间回收 | 按分配器实际占用计账，检查大块可分配槽数并触发原生淘汰；使用配套 patched Master。 |
| FlashInfer 自动调优 | 调优桶覆盖DP聚合输入：P的本地batch2048 / DP4覆盖到8192，本地dummy仍为2048。保留匹配的持久缓存及原生预热。 |
| 旧KDA检查点清理 | P的Store子配置设置 `remove_superseded_checkpoints=true`；完整新尾保存成功至少10秒后复核并非强制删除旧尾，保留租约、整块状态和原GET恢复行为。D及代码默认关闭。 |
| P 末段准入预算 | `VLLM_EXPERIMENTAL_HOT_TAIL_ADMISSION=1` 保护特定异步加载请求的 CoW 状态块余量，减少抢占重算；代码默认关闭。 |
| RDMA 目的网卡亲和 | `MC_ENABLE_DEST_DEVICE_AFFINITY=1`，结合各 GPU 分组的实际 HCA 映射。 |

## SM120 NVFP4 MoE 融合

针对 GLM 的 hidden4096 / intermediate2048 / 288 experts / top8，支持 EP4 和 EP8。内核基于 FlashInfer PR8806 的持久化调度与 P2P 协议，使用 NVFP4 E2M1、E4M3 group16 scales 和原生已校准权重，不改成 MXFP4。

- 路由、FC1、SwiGLU、重数量化、FC2 与 FP32 combine 走融合路径。
- 共享 workspace，直接引用原生权重和 scale；使用 uint4 向量读取、FC2 输入 staging 和 P 侧专家预取。
- 初始化完成 JIT、标定检查和 EP rendezvous；请求路径在当前 CUDA stream 上执行，不增加 host 同步或额外 collective。
- 分发使用各 rank 一致的 DP metadata。P dense batch 最大 192，稀疏 batch 最大 256 且总 token ≤512；D 最大 128。其他形状使用原生后端，保留原生共享专家调度。

| 环境变量 | 默认 | 4P4D P | 4P4D D |
| --- | --- | --- | --- |
| `VLLM_GLM53_NVFP4_FUSED_MOE` | 0 | 1 | 1 |
| `VLLM_GLM53_NVFP4_STAGE_FC2_INPUT` | 0 | 1 | 1 |
| `VLLM_GLM53_NVFP4_VECTOR_WEIGHTS` | 0 | 1 | 1 |
| `VLLM_GLM53_NVFP4_EXTENDED_PREFETCH` | 0 | 1 | 0 |
| `VLLM_GLM53_P_MTP_CONTEXT_ONLY` | 0 | 1 | 0 |

CUDA 源码随 wheel / sdist 安装。P2P 路径依赖 FlashInfer 0.7.1 JIT、CUDA SM120a 编译器和 Torch symmetric memory，无需 Cake 后端。环境变量在实例初始化前设置。

## P 侧 MTP context-only

专用 EP4 P 仅加载 MTP 注意力和输入融合必需的 18 个源键，跳过只用于草稿生成的 MTP FFN/专家、head 和独立 embedding。基础模型全部专家保持在 GPU。未知、重复或缺失的必需权重直接报错，并核对构造参数的完整加载覆盖。

P 仍完成 MTP 注意力和 KV 写入，在写入完成后通过原生异步事件发布交接；不生成或采样草稿。D 保留完整 MTP3，继续原生投机解码。该模式要求 V2 runner、GLM 固定几何、EP4、TP/PP/PCP/DCP1、Mooncake producer + Store，不用于 D、耦合服务、动态草稿或 EPLB。

## 4P4D 部署

4P4D 按主机计数：81–84 每台两个 EP4 P，共 8P / 32 rank；85–88 每台一个 EP8 D，共 4D / 32 rank，合计 64 GPU，TP1。

| 角色 | 参数 |
| --- | --- |
| P | 每 DP seq24 / batch2048；native auto graph 上限192；4 API/实例；KV recv4；MTP context-only。 |
| D | 每 DP seq32 / batch1024；FULL_DECODE_ONLY graph 上限128；1 API/实例；完整 MTP3。 |
| 公共 | GPU utilization 0.95，FP8 KV，prefix match256 / block64，1M 上下文，原生 async / fullISL。 |
| L2 | 83/84 各700 GiB，其余各750 GiB；建议服务预热后按本机剩余可用内存的85%设置总池容量，已有池先加回预算。 |
| Router | `router:glm53flash-1103`；P cache_aware / D round_robin；无 CPU quota，64 GiB RAM / 96 GiB RAM+swap，mems0。 |

Router 显式开启 prompt ID 复用和 length-capped 交接；不执行token容量估算或分词槽位调度，program scheduling关闭。Router 前缀树使用 LRU；字符预算与容器 RAM 上限分别配置。L3 SSD 默认关闭，可选配置见部署文档。

## 1P2D 部署

P主机两个EP4实例，D主机各一个EP8实例，共24GPU。P使用相同的NVFP4融合与MTP context-only配置；D保留完整MTP3，seq16 / batch1024 / graph64。启动命令见[deployment1p2d.md](deployment1p2d.md)，不把4P4D的D容量参数直接套入1P2D。
