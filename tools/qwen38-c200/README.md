# Qwen3.8 Flash NVFP4：复现 C200 PD 分离测试

参考运行：`run-a3e2671a9dac`。拓扑为 **3 台各 8 卡的 SM120 主机**：
P 主机跑四个 TP1/DP2/EP2，D1、D2 各跑一个 TP1/DP8/EP8。
P 和 D 共用同一个镜像，通过角色选择对应的实测 Python 源码。

## 1. 构建：任选一条路线

在 Linux amd64 构建机，从本分支仓库根目录执行。两条路线都需要原验收用的
`flashinfer_python-0.7.5+1658f0bf-py3-none-any.whl`，脚本会校验 SHA-256。
它基于 FlashInfer `1658f0bf02b65fe9a71862286f6da546d0fce890`，包含 SM120
QSA NVFP4 读写支持。请从原实验制品库获取这个 wheel 和对应的模型权重。

**通过源码构建**（需要 Docker Buildx，从原 nightly 对应源码构建 CUDA 扩展）：

~~~bash
bash tools/qwen38-c200/build.sh "$FLASHINFER_WHEEL" qwen38-nvfp4:c200
~~~

**通过基镜像构建**（推荐复现实测，保留原 nightly 二进制）：

~~~bash
docker pull vllm/vllm-openai@sha256:aeb580d55ee3dcccd7b5c78d7871326f367bee5cfd8ca8e411c253d7983a0046
bash tools/qwen38-c200/build.sh --base-image "$FLASHINFER_WHEEL" qwen38-nvfp4:c200
~~~

若主机只能访问已导入的本地 nightly，设 `QWEN38_NIGHTLY_IMAGE` 为本地镜像名再执行
同一条构建命令。脚本会校验该镜像 ID 与实测底座完全相同。

两条路线都基于 vLLM `e0d00e4b8eb0c89ec30921d0695cfdd409e9c211`，自动应用
`common.patch`、`api-protocol.patch` 与角色补丁，并在镜像构建时校验 P/D Python 文件哈希。
请以脚本准备的固定版本为实验基线。把同一镜像分发至三台机器，并记录
`docker image inspect --format '{{.Id}}' qwen38-nvfp4:c200` 的结果。

## 2. 准备模型、网络与共享参数

以下用 Bash。在各机设置相同的 `P_IP`、`D1_IP`、`D2_IP`（RDMA 可达的实际地址），
以及本机绝对路径 `MODEL_DIR`、`RUN_DIR`。三机使用同一份
`Qwen3.8-Flash-Next-NVFP4` 模型权重与 tokenizer。

从模型目录复制 `config.json` 和 `tokenizer_config.json`，分别将副本的绝对路径设为
`YARN_CONFIG` 和 `TOKENIZER_CONFIG`，然后在副本中修改以下字段：

| 文件 | JSON 字段 | 值 |
| --- | --- | --- |
| `config.json` | `text_config.max_position_embeddings` | `1048576` |
| `config.json` | `text_config.rope_parameters.rope_type` | `"yarn"` |
| `config.json` | `text_config.rope_parameters.factor` | `4.0` |
| `config.json` | `text_config.rope_parameters.original_max_position_embeddings` | `262144` |
| `tokenizer_config.json` | `model_max_length` | `1048576` |

其余字段沿用原文件，包括 mRoPE 参数和聊天模板。下方启动命令会挂载这两个副本。

~~~bash
: "${P_IP:?}" "${D1_IP:?}" "${D2_IP:?}" "${MODEL_DIR:?}" \
  "${YARN_CONFIG:?}" "${TOKENIZER_CONFIG:?}" "${RUN_DIR:?}"
export MODEL_IMAGE=qwen38-nvfp4:c200
export BASE_IMAGE=vllm/vllm-openai@sha256:aeb580d55ee3dcccd7b5c78d7871326f367bee5cfd8ca8e411c253d7983a0046
export CACHE_PREFIX="${CACHE_PREFIX:-qwen38-nvfp4-c200-fresh}"
# 在三机保持同一个本轮专用前缀；复跑换新前缀及 tokenizer 目录。
install -d "$RUN_DIR/jit" "$RUN_DIR/tokenizer-shared"
printf '%s\n' "$CACHE_PREFIX" > "$RUN_DIR/tokenizer-shared/epoch"
cat > "$RUN_DIR/mooncake.json" <<EOF
{"mode":"standalone-store","metadata_server":"http://$P_IP:18542/metadata","master_server_address":"$P_IP:18540","protocol":"rdma","global_segment_size":0,"local_buffer_size":2147483648}
EOF

rdma=(--network host --ipc host --device /dev/infiniband --cap-add IPC_LOCK
      --ulimit memlock=-1:-1 --security-opt label=disable)
model_run=("${rdma[@]}" --ulimit nofile=1048576:1048576
  -e NO_PROXY='*' -e no_proxy='*' -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e FLASHINFER_WORKSPACE_BASE=/candidate-jit
  -e VLLM_QWEN38_TOKENIZER_CACHE_GIB=0.125 -e VLLM_QWEN38_TOKENIZER_LOCAL_L1_MIB=128
  -e MOONCAKE_CONFIG_PATH=/pd/mooncake.json
  -e OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=grpc -e OTEL_TRACES_SAMPLER=parentbased_always_on
  -v "$MODEL_DIR:/model:ro" -v "$YARN_CONFIG:/model/config.json:ro"
  -v "$TOKENIZER_CONFIG:/model/tokenizer_config.json:ro"
  -v "$RUN_DIR/jit:/candidate-jit" -v "$RUN_DIR/mooncake.json:/pd/mooncake.json:ro")
serve=(/model --host 0.0.0.0 --served-model-name qwen3.8-flash-nvfp4
  --enable-prompt-tokens-details --enable-auto-tool-choice --tool-call-parser qwen3_coder
  --reasoning-parser qwen3 --enable-expert-parallel --max-num-seqs 32
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
  --max-model-len 1048576 --kv-cache-dtype nvfp4 --block-size 3200 --prefix-match-unit 3200
  --otlp-traces-endpoint "http://$P_IP:18430")
~~~

文件路径与 IP 用自己的值。端口仅暴露在隔离的测试网。实测打开 OTLP gRPC，
Collector 在 P 主机 `18430`，Router 按 `0.1` 采样。下面用同一个 nightly 运行
仓库里的接收器（通过环境变量和容器参数配置路径及 UID）。先设置 `REPO` 为本仓库绝对路径：

~~~bash
install -d "$RUN_DIR/traces"
docker run -d --name qwen38-collector --network host --memory 2g --memory-swap 2g \
  --user "$(id -u):$(id -g)" -e QWEN38_TRACE_HOST="$P_IP" -e QWEN38_TRACE_DIR=/traces \
  -v "$REPO/tools/qwen38-c200/otlp_collector.py:/collector.py:ro" \
  -v "$RUN_DIR/traces:/traces" --entrypoint python3 "$BASE_IMAGE" -B /collector.py
curl -fsS "http://$P_IP:18431/"
~~~

换接收器或关闭追踪时，记录为实验条件变化。

## 3. 启动 Mooncake Master 和各机 Store

Master 只在 P 主机启动：

~~~bash
docker run -d --name qwen38-master --network host --memory 8g --memory-swap 8g \
  --entrypoint mooncake_master "$BASE_IMAGE" \
  --rpc_port=18540 --enable_http_metadata_server=true \
  --http_metadata_server_host=0.0.0.0 --http_metadata_server_port=18542 \
  --metrics_port=18541 --logtostderr=true --eviction_high_watermark_ratio=0.98
~~~

三台各启动一个 Store。根据本机角色填 `HOST_IP`、`SEGMENT_BYTES`、`MEMORY_BYTES`、
`RDMA_DEVICES`。容量与容器内存限额沿用下表的实测值；NIC 名按本机拓扑换，
保留相同链路、NUMA 路径和 GID index。预留足够物理内存后再启动。

| 位置 | Segment bytes | 容器内存上限（bytes） | 实测设备 |
| --- | ---: | ---: | --- |
| P | 245291286528 | 262471155712 | `mlx5_4,mlx5_5,mlx5_6,mlx5_7` |
| D1 | 672039698432 | 680629633024 | `mlx5_0` 到 `mlx5_7` |
| D2 | 692523630592 | 701113565184 | `mlx5_0` 到 `mlx5_7` |

~~~bash
docker run -d --name qwen38-store "${rdma[@]}" --cap-add SYS_NICE \
  --memory "$MEMORY_BYTES" --memory-swap "$MEMORY_BYTES" \
  -e MC_GID_INDEX=3 -e MC_TCP_BIND_ADDRESS="$HOST_IP" \
  -e MOONCAKE_REQUESTER_LOCAL_HOSTNAME="$HOST_IP" \
  --entrypoint mooncake_client "$BASE_IMAGE" \
  --master_server_address="$P_IP:18540" --metadata_server="http://$P_IP:18542/metadata" \
  --protocol=rdma --host="$HOST_IP" --port=18544 --device_names="$RDMA_DEVICES" \
  --global_segment_size="$SEGMENT_BYTES" --local_buffer_size=0 \
  --enable_http_server=true --http_port=18545 --enable_offload=false --logtostderr=true
~~~

## 4. 启动 P 实例（仅 P 主机）

四个实例共享本机 20 GiB 分词缓存，各 DP 本地 L1 为 128 MiB。沿用默认 TP1，
以 `PIECEWISE` 模式启用 CUDA Graph。P 侧每个调度器的异步恢复限额为 1,048,576 token。

~~~bash
while read -r i gpu http bootstrap rpc util; do
  p_kv=$(printf '{"kv_connector":"MultiConnector","kv_role":"kv_both","engine_id":"qwen38-p%s","kv_load_failure_policy":"recompute","kv_connector_extra_config":{"connectors":[{"kv_connector":"MooncakeConnector","kv_role":"kv_producer","kv_connector_extra_config":{"mooncake_protocol":"rdma"}},{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"cache_prefix":"%s"}}]}}' "$i" "$CACHE_PREFIX")
  docker run -d --name "qwen38-p$i" --gpus "\"device=$gpu\"" "${model_run[@]}" \
    -e VLLM_QWEN38_PD_ROLE=prefill -e VLLM_HOST_IP="$P_IP" \
    -e VLLM_MOONCAKE_BOOTSTRAP_PORT="$bootstrap" -e OTEL_SERVICE_NAME="qwen38-p$i" \
    -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120 -e MAX_JOBS=4 \
    -e PYTORCH_ALLOC_CONF=per_process_memory_fraction:0.99 \
    -e VLLM_QWEN38_TOKENIZER_SHARED_GIB=20 \
    -e VLLM_QWEN38_TOKENIZER_SHARED_PATH=/tokenizer-shared/segments.sqlite \
    -e VLLM_QWEN38_TOKENIZER_CACHE_EPOCH_FILE=/tokenizer-shared/epoch \
    -v "$RUN_DIR/tokenizer-shared:/tokenizer-shared" \
    "$MODEL_IMAGE" "${serve[@]}" --port "$http" --data-parallel-size 2 \
    --data-parallel-rpc-port "$rpc" --gpu-memory-utilization "$util" \
    --max-num-batched-tokens 3200 --long-prefill-token-threshold 1600 \
    --engram-config '{"cpu_offload":true}' --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
    --kv-transfer-config "$p_kv"
done <<'PORTS'
0 0,1 18410 19710 19711 .93
1 2,3 18111 19310 19600 .94
2 4,5 18112 19320 19700 .93
3 6,7 18113 19330 19800 .94
PORTS
~~~

## 5. 启动 D 实例（D1、D2 各执行一次）

在两个 D 主机分别设置 `HOST_IP=$D1_IP` 或 `HOST_IP=$D2_IP`，以及不同的
`D_NAME`（例如 `d1`、`d2`）。启动时用与 P 相同的镜像 ID。

~~~bash
d_kv=$(printf '{"kv_connector":"MultiConnector","kv_role":"kv_both","engine_id":"qwen38-%s","kv_connector_extra_config":{"connectors":[{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"mooncake_protocol":"rdma"}},{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"cache_prefix":"%s"}}]}}' "$D_NAME" "$CACHE_PREFIX")
docker run -d --name "qwen38-$D_NAME" --gpus all "${model_run[@]}" \
  -e VLLM_QWEN38_PD_ROLE=decode -e VLLM_HOST_IP="$HOST_IP" \
  -e VLLM_MOONCAKE_BOOTSTRAP_PORT=19400 -e OTEL_SERVICE_NAME="qwen38-$D_NAME" \
  "$MODEL_IMAGE" "${serve[@]}" --port 18210 --data-parallel-size 8 \
  --data-parallel-rpc-port 19900 --gpu-memory-utilization .96 --max-num-batched-tokens 1024 \
  --engram-config '{"cpu_offload":false,"embedding_across_dp":true}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":128}' \
  --enable-force-include-usage \
  --kv-transfer-config "$d_kv"
~~~

## 6. 启动 Router（P 主机）

三接口部署使用固定的 GLM5.3 Flash Router 源码 `0966e4b11b`。构建脚本核对源码清单，
镜像记录源码提交。Router 选路、P/D 地址和 Mooncake 仍使用本页参数。

~~~bash
export ROUTER_IMAGE=router:qwen38-three-apis
bash tools/qwen38-c200/build-api-router.sh "$ROUTER_IMAGE"
~~~

原 C200 成绩来自旧 Router（镜像 ID
`sha256:764cfed279c25ff8ec808bbb929c87250161cf8e853a5e606532395d05ba3381`）与原引擎；
部署三接口版后用相同数据和计分规则重新测量。

~~~bash
docker run -d --name qwen38-router --network host --memory 32g --memory-swap 32g \
  -e NVIDIA_VISIBLE_DEVICES=void -e NO_PROXY='*' -e no_proxy='*' \
  -e VLLM_ROUTER_PD_REUSE_PROMPT_IDS=1 -e VLLM_ROUTER_PD_PREFILL_FORCE_LENGTH_CAP=1 \
  --entrypoint /usr/local/bin/vllm-router "$ROUTER_IMAGE" \
  --host 0.0.0.0 --port 18300 --prometheus-host 0.0.0.0 --prometheus-port 18301 \
  --vllm-pd-disaggregation --kv-connector mooncake \
  --prefill "http://$P_IP:18410" 19710 --prefill "http://$P_IP:18111" 19310 \
  --prefill "http://$P_IP:18112" 19320 --prefill "http://$P_IP:18113" 19330 \
  --decode "http://$D1_IP:18210" --decode "http://$D2_IP:18210" \
  --prefill-policy cache_aware --decode-policy round_robin \
  --enable-trace --otlp-traces-endpoint "$P_IP:18430" --otel-sampling-ratio 0.1
~~~

### 三接口约定

| 路由 | 客户端行为 |
| --- | --- |
| `/v1/chat/completions` | 指定、自动或必选工具实际完成时返回 `tool_calls`；预算截断返回 `length`。SSE 以 `[DONE]` 结尾；强制计量只增加最终 usage 块，显式开启连续计量时另有累计快照。 |
| `/v1/responses` | 在 `input` 提交完整历史；兼容分段助手文本、嵌套图片 URL 与包装式 function tool。冲突字段返回 400；SSE 按真实状态发送 `response.completed`、`response.incomplete` 或 `response.failed`。PD Router 按无状态接口使用，省略 `previous_response_id`（或传 null）。 |
| `/v1/messages`、`/v1/messages/count_tokens` | Messages 保留 Anthropic 工具、SSE 与原用量字段；计数只发往 P。总输入为未命中、缓存读、缓存创建之和。 |

媒体按固定 Router 的协议处理：三路支持图片，P/D 共用规范化后的输入；内联 JPEG/PNG 超过
4 MiB 时尝试压缩，视频在此部署入口返回 400。模型不存在返回相应协议的 404；上游路由不可用返回 502。
P 的内部 1-token 请求按长度结束；D 使用客户端的 stop 和输出预算。Chat 复用 P 返回的精确 token IDs，
Responses 和 Messages 复用相同请求内容，沿用 Mooncake 的 KV 交接和 P 实测缓存计量。

## 7. 验收、压测和计分

1. 确认两侧启动日志已完成 CUDA Graph 捕获、实际 KV dtype=NVFP4、物理页/PMU 均 3200。
   用 `xhigh`、thinking、stream 测普通回复和工具调用，确认 usage 完整。用约 272k token
   同前缀测冷命中 0、热命中 272000、换 P 后从 Mooncake L2 恢复 272000，并确认 D 收到 P→D KV。
   验收用不同前缀或清掉本任务验收缓存；正式开始前用新的基准前缀和空白分词缓存。
2. 按原始顺序完整使用 SWE 130k 的前 300 个会话，保留全部轮次和原始上下文。冻结输入
   `swe-first300-130k.jsonl` SHA-256
   `cb64b628c9d3b4969d6979ebe69b56c41c4758592d770d1cc98e223e80cd6c0c`，
   共 23921 计划轮。使用配套 `benchmark.example.json`，更改数据、URL 和输出路径，
   用原版 perf-benchmark 0.8.0 的环境运行。它使用 C200/U300/N300、seed 42，
   streaming、temperature 0.1、max_tokens 4096、`reasoning_effort=xhigh`、thinking 开启。
3. 剩余活跃及待执行会话总数持续低于 200 且 **严格超过 60 秒**时，向本轮压测 PID 发一次 SIGINT，
   保留 SQLite/Excel，并以该段低并发的起点作为计分截止。使用第 13157 个已完成轮次作为
   55% 起点；只取其后的完整分钟，同一成功请求 cohort 计算 TPM、输出 TPS、缓存命中率
   `sum(cached_tokens)/sum(prompt_tokens)`、TTFT、TPOT。用配套 `native_capacity_window.py`
   计算 55% 计分窗口，按剩余会话数判定截尾。
4. 参考成绩：55% 窗口 7 分钟，约 **80.23M TPM、9769 输出 tok/s、97.02% 加权命中、
   TTFT P99 4.26 s、TPOT mean 17.76 ms**。终态 17968 成功、0 失败、188 尾段取消；
   将尾段取消和请求失败分开统计；复跑时按同一口径对照参考成绩。

在新的压测目录保存已填写路径的 `benchmark.json`，并为 SQLite 和 Excel 指定新路径。
设置 `BENCHMARK_PYTHON` 为原 perf-benchmark 虚拟环境的 Python 绝对路径。使用原生入口，
并在同一进程安装上述计分扩展：

~~~bash
PYTHONPATH="$REPO/tools/qwen38-c200${PYTHONPATH:+:$PYTHONPATH}" \
  "$BENCHMARK_PYTHON" -c 'import sys; from native_capacity_window import install; from perf_benchmark.app.main import console_main; sys.argv = ["perf-benchmark", "run", sys.argv[1]]; install(); raise SystemExit(console_main())' benchmark.json
~~~

容量监控在 P 主机读取本轮 SQLite 的 `sessions`。按 `m_lifecycle_completed_ns`
排序，以第 101 个终态会话的时间戳作为剩余会话数首次低于 200 的起点。与原生进程使用
同一单调时钟，持续严格超过 60 秒后再中断。

从原实验制品库获取模型权重、冻结 SWE 数据、实测 Router 镜像和 FlashInfer wheel，
并核对哈希。镜像构建完成后，按第 7 节完成 GPU 验收，再用同一份冻结数据和计分规则
复跑 C200。
