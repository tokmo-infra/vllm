# GLM5.3 NVFP4：vLLM 0.33 跨机 DEP16

分支：`glm5.3-nvfp4-1113`。基础版本：官方 `v0.33.0`，提交
`4ba787ec9959e566d3b587a796dff07673324da5`。模型为完整的
`GlmMoeDsaForCausalLM`，不是 GLM5.3 Flash。

## 优化配置

| 配置 | 作用 | 用法 |
| --- | --- | --- |
| `VLLM_GLM53_TOKENIZER_CACHE_GIB` | 开启精确 token-ID 分段缓存；保持原生聊天模板和编码回退 | 设为 `2`；使用共享缓存时由下两项分配实际预算 |
| `VLLM_GLM53_TOKENIZER_SHARED_PATH` / `SHARED_GIB` | 同机 API 进程共享分词结果 | `/run/glm-tokenizer/segments.db`、`31` GiB |
| `VLLM_GLM53_TOKENIZER_LOCAL_L1_MIB` | 每个 API 进程的热分词缓存 | `64`；16 个 API 合计 1 GiB，加共享池共 32 GiB |
| `VLLM_GLM53_TOKENIZER_CACHE_EPOCH_FILE` | 清缓存时同步失效同机 API 缓存 | 挂载 root 所有、权限 `0600` 的 epoch 文件 |
| `lookup_async` / `load_async` | 异步查询和读回 Mooncake L2 | 均为 `true` |
| `store_compact_group_payloads` | 只存当前 KV 组实际 tensor 数据，不保存页间空洞 | `true`；使用独立缓存命名空间 |
| `--enable-flashinfer-autotune` | 启动时为实际推理内核选择配置并持久化 | 开启；调优上界覆盖 DP 汇聚后的 token 数，单 DP 的 dummy batch 不扩大 |
| `WITH_NVIDIA_PEERMEM=0` | 通过 DMA-BUF 注册 GPU RDMA 内存 | 适用于 NVIDIA open 驱动且 peer-memory 模块与 OFED 不匹配的主机；先验收 GPU 注册 |

分词缓存只接受已核验的 GLM tokenizer 指纹；改变模板参数、缓存盐或
epoch 后沿对应路径重新编码。GPU 前缀缓存和 Mooncake L2 始终独立开启。

Mooncake 保留完整 KV 组校验、PUT 压力解除后恢复、GET 租约失效后的元数据
刷新及一次重读。读回失败按实际失败处理，未确认的缓存不计成命中。

FlashInfer 使用 0.33 对应的 `0.7.2.post1`。其持久化调优 key 已包含旧版
补丁的跨 rank 修复；沿用原生多 rank 缓存一致性检查，不再次修改
FlashInfer 的 key 或把旧版本调优文件复制过来。
DP 聚合调优使用独立的 `glm53-dp-gather-v1` 缓存身份，避免加载旧的
小范围调优文件后跳过新批量的预热。

## 构建

在仓库根目录执行：

```bash
docker build -f integrations/glm53nvfp4/Dockerfile \
  --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" \
  -t vllm:glm5.3-nvfp4-1113 .
```

镜像基底固定为官方 `vllm/vllm-openai:v0.33.0` 的 digest。安装器先检查
vLLM 正式版本和每个待修改文件的 0.33 原始指纹，再安装清单内的 Python
文件；保留该正式版的 Torch、CUDA、FlashInfer 和原生扩展。发布清单位于
`/opt/glm53-nvfp4-release.json`。旧 0.32、0.33 预发布版和混合源文件会
在构建时被拒绝。

## 两机耦合实例

每台 8 GPU，TP1、全局 DP16/EP16。头节点运行 API，第二节点 headless；
替换以下地址、网口和模型路径为现场实际值。所有权重和完整 MTP 在 GPU。
不设置 `--attention-backend`、`--moe-backend` 或 `--all2all-backend`，由
0.33 原生逻辑选择并记录实际后端。

每台独立启动 750 GiB Mooncake Store，共用一个 Master，Master 淘汰
水位 `0.98`；配置文件的 `global_segment_size` 在模型进程中设为 `0`，
避免模型进程再次分配 750 GiB。每台模型的 `MOONCAKE_CONFIG_PATH`
指向本机配置，`local_buffer_size` 设为 `2147483648`。

```json
{
  "mode": "standalone-store",
  "metadata_server": "http://MASTER_IP:62242/metadata",
  "master_server_address": "MASTER_IP:62241",
  "protocol": "rdma",
  "device_name": "mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7",
  "global_segment_size": 0,
  "local_buffer_size": 2147483648,
  "enable_offload": false
}
```

模型公共参数：

```bash
vllm serve /model --served-model-name glm53-nvfp4 \
  --language-model-only --tensor-parallel-size 1 \
  --data-parallel-size 16 --data-parallel-size-local 8 \
  --data-parallel-address HEAD_IP --data-parallel-backend mp \
  --data-parallel-rpc-port 20550 --enable-expert-parallel \
  --enable-ep-weight-filter --max-model-len 131072 \
  --max-num-seqs 32 --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.90 --kv-cache-dtype fp8 --block-size 64 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-prefix-caching --scheduler-reserve-full-isl \
  --enable-flashinfer-autotune --enable-cumem-allocator --numa-bind \
  --reasoning-parser glm47 --enable-auto-tool-choice --tool-call-parser glm47 \
  --enable-prompt-tokens-details --enable-force-include-usage \
  --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","engine_id":"UNIQUE_NODE_ID","kv_load_failure_policy":"fail","kv_connector_extra_config":{"cache_prefix":"glm53-v033-dep16","lookup_async":true,"load_async":true,"store_compact_group_payloads":true}}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":128}'
```

头节点额外添加 `--host HEAD_IP --port 19483 --api-server-count 16`；第二
节点添加 `--headless --data-parallel-start-rank 8`。两机分别设置本机
`VLLM_HOST_IP`、`MC_TCP_BIND_ADDRESS`、`MOONCAKE_REQUESTER_LOCAL_HOSTNAME`
及实际 `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME`。RDMA GID、HCA 列表及
GPU 亲和以实测拓扑为准。模型进程保留 `VLLM_USE_V2_MODEL_RUNNER=1`。

在 CUDA 可见显存约 83 GiB 的 6000D 上，完整模型每 rank 权重约
60.33 GiB，CUDA 图约 0.95 GiB。上述配置的 GPU KV 约 8.8 GiB/rank。
FlashInfer 调优还会临时分配数 GiB 工作区，因此首次启动从 `0.90` 开始，
将缓存目录持久挂载。调优完成后的运行峰值与 KV 容量分别核验，再调整
利用率；保留异步、MTP 和 CUDA 图，不靠关闭这些功能节省显存。

Router 使用 GLM 分支的缓存亲和及低并发均衡实现，耦合部署设置
`--worker-urls http://HEAD_IP:19483 --intra-node-data-parallel-size 16`
和 `--policy cache_aware`。32 GiB 是 API 层分词缓存预算，Router 的
`--max-tree-size` 只控制其前缀路由树。

## 验收与记录

先核版本、权重结构、实际后端及全部 16 rank；再验证精确分词、首次
冷请求、本地前缀命中、真实 L2 读回，以及改变输入后的图重放输出。
随后以真实 SWE 完整前轮对话组成约 100k token 输入，运行全局 C32，
分别记录各缓存条件下的 TTFT/TPOT/Decode Mean、P50、P90、P99、
实际输入输出 token、Prefill token/s、L2 读取字节/耗时、吞吐和失败数。

Flash 版 H4096/E288、单机 EP4/EP8 的专用融合 kernel 不作为完整模型
H6144/E256、跨机 EP16 的已适配实现。先以原生支持路径验证，再单独
验收完整模型的数值、通信协议和性能。耦合服务保留完整 MTP，不启用
只用于 P 角色的 MTP 权重精简。
