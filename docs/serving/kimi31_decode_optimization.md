# Kimi K3.1 decode 优化：TP/EP 共享输入复用与计算重叠

本文说明 Kimi K3.1 Hybrid decode 路径上的共享专家输入复用和计算重叠。**已选择 `decode_backend=latency_specialized`，且实际 MegaMoE/shared 布局兼容时默认自动启用。通用部分不要求 DCP8、DP1、DSpark 4 步或 C64，也不固定调度容量和 shared 权重尺寸。** 专用后端的原有支持范围不变，不会强制切换后端。

## 默认行为与关闭方式

适用路径通过实际 `HybridSharedMoE` provider 调用，分开选择通用调度与特定形状的内核调优：

- 复用 Hybrid 在 routed SP 切片前已有的完整 BF16 hidden，省去 shared 分支随后重复的输入 all-gather。
- 将 shared expert 的原 BF16 算子与 routed MegaMoE 计算重叠。
- 仅当 routed 输入满足独立的已测形状条件时，追加 MegaMoE staged tile 调度；其他形状仍可使用前两项优化。

保留 routed SP/EP、shared reduce-scatter、routed 输出变换及最终输出 all-gather；不是用不完整 shard 替代 shared 输入，也不改变 shared 算子、dtype 或模型权重。

沿用实际加载的 shared 布局，不强改权重。如果使用既有的 TP 分片设置：

```bash
export VLLM_KIMI_K3_SHARD_SP_SHARED_EXPERT=1
```

则复用完整输入并保留 shared RS。该变量为0、shared 权重复制时，也可重叠计算，但输入使用本 rank 的原 SP 局部行，不额外加 RS，也不声称省去了原本不存在的 shared 输入 AG。

已选专用后端时，可省略 `hybrid_shared_overlap`，由适用条件自动选择。回退/对照配置为：

```json
{
  "decode_backend": "latency_specialized",
  "hybrid_shared_overlap": false
}
```

这是 `additional_config` 的现有配置入口，不是新的服务命令或 CLI 开关。`false` 禁用本组合 provider，恢复原 Hybrid MoE 路径；**不代表切回 generic 模型实现**。

## 通用适用条件与回退

| 项目 | 通用复用/重叠的选择方式 |
| --- | --- |
| 并行布局 | 继承 Hybrid 的 TP/EP 支持；TP 组属于对应 EP 组，EP 可跨多个 DP 组，不要求三组 ranks 相同 |
| DCP | 此 MoE 优化不检查 DCP 数；上游 attention 仍只支持 DCP1 或 DCP=TP，不能据此宣称任意 TP/DCP 组合可运行 |
| q / 并发 | 读取实际 `runtime.q` 与已规划的捕获尺寸，不固定 q5、C64 或 M320；无投机的 q1 仍受上游 Hybrid 规划限制 |
| 容量与维度 | 使用实际专家缓冲容量和 gate、latent、shared 权重维度，不固定 maxbatch16384、local experts112 或 shared intermediate768 |
| GPU | 继承原专用后端/MegaMoE 的硬件要求；通用部分不额外限定 B300/148SM |
| shared 布局 | 支持原 TP 分片或 SP 局部复制权重；BF16 原投影及原 activation，side stream 只运行纯计算，按实际布局决定是否 shared RS |
| 不兼容配置 | 原有 DBO、breakable capture 或不满足实际布局条件的路径回退；不强开不兼容 collective |

例如 TP4/DP2/EP8 的两个 TP 组各自复用自己的完整输入，routed 仍调用原专家入口，shared 仍在各自 TP4 内 RS；不是把不同 DP 的输入混合。TP8/DCP1 与 TP8/DCP8 不再因 DCP 数被区别对待。其他并发或 DSpark 步长也不再仅因这些数值回退。

输入复用的契约来自 Hybrid：每个 TP rank 的完整 hidden 与原 `AG(shard(full))` 的有效行一致。M 不能整除 TP 时，先将完整输入补零到 `ceil(M/TP)*TP` 行，再运行 shared GEMM，与原 all-gather 输入形状一致。只补人工 SP 尾行，不清零原始输入中由请求 padding mask 标记的行；即使某个 rank 只有 padding，也照常参与原 collective。

上述补齐/RS只用于真实 shared TP 分片。权重复制时仍只计算本地 SP 行；不能因为另一种布局用了 full hidden，就将其套到复制权重上。routed 普通分支调用原专家入口，保留其量化格式、EPLB映射/计数及回调；这不扩展上游模型本身的功能支持声明。

通用分支不自行初始化新的 EP symmetric buffer 或首发同步，routed 继续交给原专家对象；shared side stream 不放 EP/TP collective。原 routed EP → shared TP RS → 最终 TP AG 的顺序保留。TP1/DEP 没有这个重复 TP 输入 AG，而且使用不同 executor；EP 关闭也走不同 executor，本改动不冒充这些路径已经接入该 provider。prefill 不在本次范围内。

### 独立的 staged tile 条件

40→72 行的 routed padding 是内核调度策略，不是通用 provider 的启用门槛。只在 B300/148SM、同一 TP8/EP8 组、实际局部40行/全组320行、原 MXFP4 Kimi 权重布局/容量及 native `BLOCK_M(40)=16`、`BLOCK_M(72)=32` 条件下使用。这里按实际 MoE 行数判断，不绑定 DSpark 步长或请求数；相同 M320 可以来自不同请求数和 q 的组合。其他形状调用原专家 forward，同时仍可执行共享输入复用和计算重叠。

tile 路径保留原量化、scales、activation、`fast_math`、kernel kwargs 和 capture 回调，每次重置无效尾路由。不能把填更多行当作任意 batch 都会更快的依据。

**代码适用范围与实测证据分开：以下性能和字节一致结果只来自已测的 TP8/DP1/DCP8、q5、C64。新增 TP/DP、其他 q 和尾 padding 的 GPU 回归尚未执行，不能把通用化等同于这些配置已验收或保证零性能回退。**

## 实测结果与计时边界

2026 年 11 月 13 日的最终 R10 短模拟使用真实 target/draft 权重、C64q5、40960 KV 长度元数据。每臂汇总 12 个样本，每个样本先取所有 rank 的最大 GPU cycle 时延：

| 路径 | cycle 中位数 |
| --- | ---: |
| 原版 stock Hybrid | 54106.290 us |
| shared 输入复用/重叠 + staged tile | 49835.375 us |

中位数下降约 7.89%；stock 首尾从 54503.74 us 变为 53677.50 us，存在约 1.52% 的时序漂移，因此保守表述为 **约 7–8% 的所测 GPU cycle 下降**，不声称消除了漂移或已得到生产吞吐置信区间。

若假定每 cycle 的 5 个 token 全部接受，可换算为单路约 92.4 → 100.3 token/s、64 路总计约 5914 → 6421 token/s。**这只是全接受固定状态模拟的算术换算，不是自然接受 TPOT、真实请求吞吐或 API 服务实测。**

计时包含 target + draft forward 和草稿 Markov sampling；不包含 target sampling、rejection、CPU 调度、请求队列或完整 SWE 工作负载。历史 KV 零初始化，按同一 benchmark 快照恢复，不是实际生成 40K 历史后的连续自然 decode。

### 正确性已经证明到哪里

- 全部 8 rank 上，跨臂 target 最终 hidden、5 个 aux tensor 的原始字节一致，返回草稿 token 相同。
- 这支持“所测优化相对原 stock Hybrid 未新增这些输出差异”，不证明草稿 logits cache、全部后续 recurrent/KV 状态或自然 rejection 等价。
- **原版 generic 与原版 Specialized/Hybrid 的绝对输出差异仍未解释。** 路径不同不是精度豁免；不能把跨优化臂字节一致写成模型绝对精度或模型质量已验收。
- 尚未验证自然接受率、长期多 step 演化、真实服务或完整任务质量；本次未新增服务补测。

### 未合入的候选

KDA CTA148 叠加后 cycle 为 49792.54 us，增益仅约 0.086%，不足以支持合入；保留原 KDA。shared FC1 融合候选约慢一倍，也不合入。这些不是默认优化的一部分。

## 内存、镜像与部署限制

benchmark 为 KV 分配了每卡 24 GiB，并保存/恢复所需状态快照。该值仅描述此次 benchmark 的 KV/快照实验条件；不是默认生产 KV 预算、每个请求需要 24 GiB，或新增的服务内存配置要求。

本次在既有镜像及源代码覆盖环境中完成验证，**没有构建或发布新镜像**。Git 源码的默认行为变化不代表正在运行的服务已更新；本次没有启动、更新或验收真实 API 服务。

## 测试与尚未覆盖的 CI

回归测试扩展在现有 `tests/models/kimi_k3/nvidia/specialized_decode/test_kimi_k3_sdec_hybrid.py`，复用其 ForwardBench → `LLM` 初始化、worker、`admit()` 配置适配和实际 capture metadata 记录。没有新增 Mock、手工伪造 provider 或多进程 harness；保留原测试的 config-only/dummy-weight fixture，不能把它写成真实权重加载或模型精度测试。真实权重依据是前述 R10，而不是新增 pytest。

保留一次 worker/admit/check，不增加第二次模型加载。测试覆盖 M<TP、非整除尾 padding 和多种 capture rows，断言实际 provider 类型、shared 输入/partial 补齐形状以及人工尾行归零；不再要求只有 M320 进入优化。显式 `false` 用真实 config 的 `dataclasses.replace` 检查 factory 早返回，不修改全局 runtime。原 graph/eager 的 `5e-3` 阈值不变。

既有默认 TP4 测试也应进入通用路径；设置 `KIMI_K31_HYBRID_WORLD=8`、`KIMI_K31_HYBRID_TP=4`，并将 `KIMI_K31_HYBRID_DCP` 分别设为1和4可覆盖 TP4/DP2/EP8。`KIMI_K31_HYBRID_DRAFT_TOKENS` 控制真实 DSpark 测试步长，例如2对应q3，各 case 和 capture 尺寸随之推导。`KIMI_K31_HYBRID_SHARED_TP=0` 选择复制 shared 权重的测试。它们是测试变量，不是新增运行时优化开关。

当前 Buildkite 没有可用的 NVIDIA Kimi 专用测试资源/选中该覆盖的任务，不能把未运行写成 CI 已通过。没有新增 skip、xfail、重试或放宽数值阈值。已有硬件条件仍需在解释测试结果时考虑。

GPU 优化和测试在 2026 年 11 月 13 日 16:40 CST 截止；此后按用户要求做默认启用和通用化源码调整、本地静态检查，未重启服务或补跑 GPU 测试。R10 不等于新增布局回归已通过，也不代表当前镜像包含最终源码。
