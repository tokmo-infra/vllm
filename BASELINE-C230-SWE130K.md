# C230 · SWE13w · 原始模型基线

## 拓扑与镜像

- 82：两个 TP1/DP4/EP4 P，分别使用 GPU0–3、GPU4–7。
- 83、84：各一个 TP1/DP8/EP8 D。
- 81：停止本任务的模型及 Mooncake Store，其他任务不动。
- 模型使用原 C230 验收记录中的实际镜像 ID：sha256:30bd12541a5b98920ce69a6ad3d270c5f68e15caa59447358bd7b4434883500c，不使用后来被覆盖的同名 tag。

## 模型参数

| 项目 | P | D |
| --- | --- | --- |
| max-num-seqs／DP | 24 | 16 |
| max-num-batched-tokens／DP | 2048 | 1024 |
| gpu-memory-utilization | 0.95 | 0.95 |
| speculative-config | method=mtp，num_speculative_tokens=3 | 同左 |
| compilation-config | 省略，恢复原默认 | FULL_DECODE_ONLY，max_cudagraph_capture_size=48 |
| KV | fp8，prefix-match-unit=256 | 同左 |
| max-model-len | auto | auto |
| kernel-config | enable_flashinfer_autotune=false | 同左 |

P 额外保留用户指定的 VLLM_MOONCAKE_LOAD_RECV_THREADS=4。

不启用 prefill_only_target_state、mamba_current_state_only、release_finished_mamba_scratch、remote_mamba_speculative_blocks，不覆盖为 proposal=0 或 graph=NONE。

其他参数使用服务器保存的原 C230 启动命令，包括原生后端选择、NUMA、网卡和 GPU 分组。每次启动生成新的 engine_id。

## Router 与 Mooncake

- Router 保留用户指定的新 P 计算量／D 请求数与 KV 容量路由，配置见 Router 仓库根目录 PD-TOKEN-BALANCE.md。
- Router 分词最多四路并行，容器内存 8GiB。
- Mooncake 保留 82/83/84 各 750GiB，共 2250GiB，水位 98%；启动前清空本任务旧 KV。
- Store 沿用原 C230 的 load_async=true、store_compact_group_payloads=true，不额外设置 lookup_async。

## 压测

C230／345 用户／345 样本，SWE13w 前 345 条完整自然轨迹。55% 后完整分钟统计，持续 60 秒不能维持目标并发后收尾；不采样 Nsight，不按 TTFT 止损。

原价估算收入和乘 0.2 后收入都保留，原生 DB 留服务器。
