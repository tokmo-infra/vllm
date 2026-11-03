# vLLM 0.32.0 / Router 0.1.16 · GLM-5.3 Flash

本分支集成GLM-5.3 Flash的推理、缓存与路由优化。以下按功能列出对应提交；启动命令和镜像版本见 [1P2D部署文档](DEPLOYMENT.md)。

## 通用改进

| 功能 | 对应提交 | 适用范围 |
| --- | --- | --- |
| FlashInfer MoE调优：统一各rank的调优键，支持按DP/rank加载预调优配置。 | `e79546dc` | FlashInfer CUTLASS / CuTe MoE。 |
| 请求阶段计时：记录排队、prefill和首输出等阶段，标记缺失事件。 | `771b4aca` | 请求统计与性能分析。 |
| Mooncake GET租约刷新：租约过期后重新查询副本并重试一次，记录失败与重试指标。 | `14c6dae4` | Mooncake Store读取。 |
| Mooncake PUT恢复：分配压力缓解后，限频探测并恢复后续缓存保存任务。 | `7a26c0a8` | Mooncake Store发送线程。 |

## 模型与硬件适配

| 功能 | 对应提交 | 适用范围 |
| --- | --- | --- |
| SM103 / SM120 NVFP4：补齐原生扩展编译路径和算子注册。 | `40d41212` | B300、RTX PRO 6000及对应NVFP4后端。 |
| FP8 MLA紧凑布局：支持528字节KV行及配套读写、传输描述。 | `40d41212`、`e79546dc` | 对应模型维度和后端支持的FP8 MLA。 |
| MTP元数据复用：共享top-k时，后续draft步骤跳过冗余索引元数据处理。 | `40d41212`、`e79546dc` | GLM原生MTP路径。 |
| 混合KV紧凑存储：按组保存有效KV内容，减少stride空洞，并隔离不同编码的缓存命名空间。 | `40d41212`、`e79546dc` | Mooncake混合KV缓存。 |
| 混合前缀与分阶段查询：按组确认可加载前缀，逐步查询Mamba边界，支持异步查询和加载。 | `40d41212`、`e79546dc` | 混合KV组与前缀复用。 |
| 共享Tokenizer缓存：同机共享精确分词结果，每API保留私有L1；通过模型哈希、Tokenizer哈希和epoch管理缓存。 | `70902bd6`、`dec0a242` | GLM Tokenizer与同机多API进程。 |

## 缓存容量与调度

| 功能 | 对应提交 | 适用范围 |
| --- | --- | --- |
| Mooncake目的HCA亲和：为传输选择匹配的目的端网卡。 | `5706f4e4`（配置） | Mooncake SDK / RDMA。 |
| Master实际占用计账：按分配器实际占用空间计算缓存水位，使淘汰时机与真实容量一致。 | `43cb7086` | Mooncake Master的CPU缓存内存分配。 |
| Master大块空间检查：可容纳目标大小KV的空闲槽不足时，提前淘汰符合条件的旧缓存，减少大块保存失败。 | `5e9ff92b` | Mooncake CPU缓存；沿用原生租约、pin和引用保护。 |
| P端末块CoW准入预算：为异步加载请求预留后续Mamba状态块，减少末段容量不足触发的抢占。 | `4b19117a` | 生产者异步加载、同一物理末块等适用条件；见 [调度开关说明](docs/design/hot_tail_admission.md)。 |

## Router

对应提交位于配套的 `tokmo-infra/router` 同名GLM分支；本仓库通过 `e79546dc` 集成。

| 功能 | Router提交 | 适用范围 |
| --- | --- | --- |
| 请求负载与usage：维护D端预占负载，处理流式结束和取消，保留历史前缀及usage信息。 | `280e8ff`、`92c2ac6` | PD请求生命周期与统计。 |
| PD容量路由：结合P端计算量、请求数和KV容量分配负载，并记录前置处理耗时。 | `53e0758`、`998b908` | PD分离部署。 |
| 耦合实例路由：按实例内各DP的KV与负载选择目标。 | `48740d2` | 耦合实例部署。 |

## 开关与配置项

| 配置项 | 默认值 | 当前1P2D配置 |
| --- | --- | --- |
| 预调优只加载 `GLM_FLASHINFER_PRETUNED_ONLY` | 未设置 | 未设置，使用原生自动调优。 |
| Store紧凑布局 `store_compact_group_payloads` | `false` | `true`。 |
| Store异步加载 `load_async` | `true` | `true`。 |
| Store异步查询 `lookup_async` | `false` | `true`。 |
| Tokenizer缓存容量 `VLLM_GLM53_TOKENIZER_CACHE_GIB` | `0` | P为`2`、D为`16` GiB/API。 |
| Tokenizer共享池 `VLLM_GLM53_TOKENIZER_SHARED_PATH` | 空 | `/run/glm-tokenizer/segments.db`。 |
| 目的HCA亲和 `MC_ENABLE_DEST_DEVICE_AFFINITY` | 未设置 | P设为`1`。 |
| Master目标大块大小 `MC_EVICTION_FIT_REQUEST_BYTES` | `0`，关闭 | `60318720`字节。 |
| Master空闲槽阈值 `MC_EVICTION_FIT_MIN_SLOTS` | `64` | `64`。 |
| Master检查间隔 `MC_EVICTION_FIT_CHECK_MS` | `1000` | `1000`毫秒。 |
| Router PD路由 `VLLM_ROUTER_PD_TOKEN_BALANCE_CONFIG` | 未设置 | `/run/token-balance.json`。 |
| Router耦合路由 `VLLM_ROUTER_COUPLED_KV_BALANCE` | 未设置 | 关闭。 |
| P末块准入预算 `VLLM_EXPERIMENTAL_HOT_TAIL_ADMISSION` | `0`，关闭 | 关闭；设为`1`开启。 |
| P末块预算进度日志 `VLLM_HOT_TAIL_ADMISSION_DIAGNOSTICS` | `0`，关闭 | 关闭；设为`1`记录有限进度事件。 |

## 相关文档

- [1P2D部署与启动命令](DEPLOYMENT.md)
- [源码与构建](GLM53FLASH-TECHNICAL.md)
- [FlashInfer预调优](FLASHINFER-AUTOTUNE.md)
- [集成版本与补丁](integrations/glm53flash/manifest.json)
- [Master构建配置](integrations/glm53flash/mooncake-master.json)
