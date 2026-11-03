# GLM5.3 Flash 1103

本分支提供GLM-5.3 Flash推理、共享Tokenizer、Mooncake缓存恢复和HCA部署配置。源码基于实际运行版本5df63a720，功能按提交说明，启动命令统一见[DEPLOYMENT.md](DEPLOYMENT.md)。

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

## 4. 目的网卡亲和与部署

P启动时设置MC_ENABLE_DEST_DEVICE_AFFINITY=1，为SDK选择目的端HCA提供亲和提示；GPU到本地HCA的绑定继续按对应GPU分组填写。推理示例使用TP1、P双EP4 / D各EP8，P每DP seq24/batch2048、D每DP seq16/batch1024、每台750GiB Store，原生MTP3与异步调度，Tokenizer共享池加L1为16GiB/台。

源码与镜像指纹集中在integrations/glm53flash/release.json；镜像构建、启动命令和可选配置集中在DEPLOYMENT.md。
