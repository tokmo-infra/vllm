# EP4 P / EP4+EP8 D · C288

## 拓扑与参数

| 主机 | 实例 | GPU / CX8 HCA | 每DP seq / batch tokens | CUDA Graph |
| --- | --- | --- | --- | --- |
| 81 | P0：TP1/DP4/EP4 | GPU0—3 / mlx5_0—3 | 24 / 2048 | NONE |
| 81 | P1：TP1/DP4/EP4 | GPU4—7 / mlx5_4—7 | 24 / 2048 | NONE |
| 82 | P0：TP1/DP4/EP4 | GPU0—3 / mlx5_0—3 | 24 / 2048 | NONE |
| 82 | D1：TP1/DP4/EP4 | GPU4—7 / mlx5_4—7 | 24 / 1024 | FULL_DECODE_ONLY，128 |
| 83 | D0：TP1/DP8/EP8 | GPU0—7 / mlx5_0—7 | 24 / 1024 | FULL_DECODE_ONLY，128 |
| 84 | D0：TP1/DP8/EP8 | GPU0—7 / mlx5_0—7 | 24 / 1024 | FULL_DECODE_ONLY，128 |

- 公共：GLM-5.3 Flash NVFP4，util0.95，FP8 KV，prefix256，block-size64，max-model-len auto。
- P 保留全局MTP3/draft-prefill，proposal固定0，目标验证草稿槽0；D保持正常MTP3。
- P接收线程4；P/D均绑定对应CX8，GID3；MoE/Attention/Linear后端自动，FlashInfer自动调优关闭。
- 81—84各750GiB Mooncake Store，共3000GiB；水位98%，local_first。
- Router：P/D cache_aware，负载阈值4/1.5；12路P、20路D。
- SWE8w：C288/U432/N432，55%后完整分钟统计，持续60秒并发不足收尾，无Nsight/TTFT止损。

旧C230命令见 [DEPLOYMENT-C230.md](DEPLOYMENT-C230.md)，源码与历史定位见 [GLM53FLASH-TECHNICAL.md](GLM53FLASH-TECHNICAL.md)。

## 1. P-only镜像

当前镜像 native-indexer-v4 已移除实验性 SM120 融合索引器及共用开关；Prefill / Decode 使用原生索引器。

本轮镜像包含 integrations/glm53flash/experiments/prefill-only.patch，
同时修改 config/vllm.py、KDA、Mooncake connector、scheduler 和 GPU model runner。
DP只放行覆盖所有合法批量、固定为0的P-only proposal，其他动态schedule仍沿用原生DP保护。空闲DP的dummy路径同样使用0 proposal，保留draft-prefill并对齐EP collective。

~~~bash
python3 integrations/glm53flash/prepare_sources.py sources-p --prefill-only-experiment
cat > /tmp/Dockerfile.glm-p-only <<'EOF'
FROM yyf-glm53flash:fp8-wire-v1
COPY sources-p/vllm/vllm/config/vllm.py /usr/local/lib/python3.12/dist-packages/vllm/config/vllm.py
COPY sources-p/vllm/vllm/models/glm5next/common/kda.py /usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/common/kda.py
COPY sources-p/vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py
COPY sources-p/vllm/vllm/v1/core/sched/scheduler.py /usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py
COPY sources-p/vllm/vllm/v1/worker/gpu/model_runner.py /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_runner.py
COPY sources-p/vllm/vllm/models/glm5next/nvidia/sparse_indexer.py /usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/sparse_indexer.py
COPY sources-p/vllm/vllm/v1/attention/backends/mla/indexer.py /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/indexer.py
COPY sources-p/flashinfer/flashinfer/dsa_indexer.py /usr/local/lib/python3.12/dist-packages/flashinfer/dsa_indexer.py
RUN rm -f /usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/sm120_dsa_adapter.py /usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/sm120_dsa_prefill.py && rm -rf /usr/local/lib/python3.12/dist-packages/flashinfer/experimental/sm120_dsa_indexer /usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/__pycache__ /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/__pycache__ /usr/local/lib/python3.12/dist-packages/flashinfer/__pycache__
EOF
docker build -f /tmp/Dockerfile.glm-p-only -t yyf-glm53flash:native-indexer-v4 .
~~~

各模型主机设置：

~~~bash
export IMAGE=yyf-glm53flash:native-indexer-v4
export ROUTER_IMAGE=local/glm53flash-router:pd-reservation-candidate-20261027
export MODEL_DIR=/data/models/GLM-5.3-Flash-NVFP4
export WORK=/data/bench/eng-yangyufei/glm53flash-pd-828384-20261027/integration-20261028/ep4p-ep4ep8d-c256-20261029
export NODE_ID=81  # 按主机改为81/82/83/84
export NODE_IP="11.8.0.$NODE_ID"
export STORE_PREFIX=glm53flash-ep4p-ep4ep8d-c256-20261029-prefix256
export API_KEY  # 从运行环境的私有配置读取
mkdir -p "$WORK"
~~~

## 2. Mooncake

本轮master和四个750GiB Store全冷重启。Store容器上限896GiB，池大小仍750GiB。
以下命令用于空环境；83启动master：

~~~bash
docker run -d --name glm-mooncake-master --network host --runtime=runc \
  -e NVIDIA_VISIBLE_DEVICES=void --entrypoint mooncake_master "$IMAGE" \
  --rpc_port=61241 --enable_http_metadata_server=true \
  --http_metadata_server_host=0.0.0.0 --http_metadata_server_port=61242 \
  --metrics_port=61243 --eviction_high_watermark_ratio=0.98 \
  --allocation_strategy=local_first --logtostderr=true
~~~

81—84各启动一个Store：

~~~bash
LIBCUDA_PATH=$(readlink -f /usr/lib/x86_64-linux-gnu/libcuda.so.1)
docker run -d --name "glm-mooncake-store-$NODE_ID" --network host --runtime=runc \
  --memory 896g --memory-swap 896g --device /dev/infiniband \
  --cap-add IPC_LOCK --cap-add SYS_NICE --ulimit memlock=-1:-1 \
  -v "$LIBCUDA_PATH:/opt/host-driver/libcuda.so.1:ro" \
  -e LD_LIBRARY_PATH=/opt/host-driver:/usr/local/cuda/lib64 \
  -e NVIDIA_VISIBLE_DEVICES=void --entrypoint mooncake_client "$IMAGE" \
  --master_server_address=192.168.17.83:61241 \
  --metadata_server=http://192.168.17.83:61242/metadata \
  --protocol=rdma --host="$NODE_IP" --port=61244 \
  --device_names=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7 \
  --global_segment_size=805306368000 --local_buffer_size=0 \
  --enable_http_server=true --http_port=61245 --enable_offload=false --logtostderr=true
~~~

## 3. P/D模型启动函数

P新增四个字段位于 MooncakeConnector.kv_connector_extra_config，不是MultiConnector顶层或Store内。
decode缓存保存不覆盖，沿用原生默认false。每次启动使用新的engine_id，避免D沿用上次P进程的KV连接元数据。

~~~bash
launch_instance() {
  ROLE="$1"; NAME="$2"; GPUS="$3"; DP="$4"; PORT="$5"; BOOT="$6"; RPC="$7"
  HCAS=$(printf '%s' "$GPUS" | sed 's/\([0-9]\)/mlx5_\1/g')
  IB_HCA="=$(printf '%s' "$HCAS" | sed 's/,/:1,/g'):1"
  DIR="$WORK/$NAME"; mkdir -p "$DIR/cache"
  ENGINE_ID="$NAME-$(date +%s%N)"
  cat > "$DIR/mooncake.json" <<EOF
{"mode":"standalone-store","metadata_server":"http://192.168.17.83:61242/metadata",
 "master_server_address":"192.168.17.83:61241","protocol":"rdma","device_name":"$HCAS",
 "global_segment_size":0,"local_buffer_size":2147483648,"enable_offload":false,"tenant_id":"default"}
EOF
  if [ "$ROLE" = P ]; then
    SEQ=24; BATCH=2048; RECV=4
    DIRECT_ROLE=kv_producer; STORE_ROLE=kv_both
    DIRECT_EXTRA=',"prefill_only_target_state":true,"mamba_current_state_only":true,"release_finished_mamba_scratch":true,"remote_mamba_speculative_blocks":3'
    SPEC='{"method":"mtp","num_speculative_tokens":3,"num_speculative_tokens_per_batch_size":[[1,64,0]]}'
    GRAPH='{"cudagraph_mode":"NONE"}'
  else
    SEQ=24; BATCH=1024; RECV=1
    DIRECT_ROLE=kv_consumer; STORE_ROLE=kv_consumer; DIRECT_EXTRA=''
    SPEC='{"method":"mtp","num_speculative_tokens":3}'
    GRAPH='{"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":128}'
  fi
  KV=$(cat <<EOF
{"kv_connector":"MultiConnector","kv_role":"kv_both","engine_id":"$ENGINE_ID",
 "kv_load_failure_policy":"fail","kv_connector_extra_config":{"connectors":[
 {"kv_connector":"MooncakeConnector","kv_role":"$DIRECT_ROLE",
  "kv_connector_extra_config":{"mooncake_protocol":"rdma","device_name":"$HCAS"$DIRECT_EXTRA}},
 {"kv_connector":"MooncakeStoreConnector","kv_role":"$STORE_ROLE",
  "kv_connector_extra_config":{"cache_prefix":"$STORE_PREFIX","load_async":true,
   "lookup_async":true,"store_compact_group_payloads":true}}]}}
EOF
)
  docker run -d --name "$NAME" --init --network host --shm-size 64g \
    --memory 96g --memory-swap 96g --gpus "\"device=$GPUS\"" \
    --device /dev/infiniband --cap-add IPC_LOCK --cap-add SYS_NICE \
    --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 \
    -v "$MODEL_DIR:/model:ro" -v "$DIR:/work" -v "$DIR/cache:/root/.cache" \
    -e VLLM_HOST_IP="$NODE_IP" -e NCCL_SOCKET_IFNAME=enp115s0f1np1 \
    -e GLOO_SOCKET_IFNAME=enp115s0f1np1 -e NCCL_IB_HCA="$IB_HCA" \
    -e NCCL_IB_GID_INDEX=3 -e MC_GID_INDEX=3 -e NCCL_DEBUG=INFO \
    -e VLLM_SERVER_DEV_MODE=1 -e VLLM_USE_V2_MODEL_RUNNER=1 \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e MOONCAKE_CONFIG_PATH=/work/mooncake.json -e WITH_NVIDIA_PEERMEM=0 \
    -e VLLM_MOONCAKE_BOOTSTRAP_PORT="$BOOT" -e MALLOC_ARENA_MAX=2 \
    -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120 -e VLLM_MOONCAKE_LOAD_RECV_THREADS="$RECV" \
    --entrypoint vllm "$IMAGE" serve /model \
    --served-model-name glm-5.3-flash-nvfp4 --host "$NODE_IP" --port "$PORT" \
    --api-key "$API_KEY" --api-server-count 1 --language-model-only \
    --tensor-parallel-size 1 --data-parallel-size "$DP" --data-parallel-size-local "$DP" \
    --data-parallel-backend mp --data-parallel-rpc-port "$RPC" \
    --enable-expert-parallel --enable-ep-weight-filter \
    --max-model-len auto --max-num-seqs "$SEQ" --max-num-batched-tokens "$BATCH" \
    --gpu-memory-utilization 0.95 --kv-cache-dtype fp8 \
    --block-size 64 --prefix-match-unit 256 --mamba-cache-mode align \
    --speculative-config "$SPEC" --compilation-config "$GRAPH" --kv-transfer-config "$KV" \
    --scheduler-reserve-full-isl --enable-prefix-caching --reasoning-parser glm47 \
    --enable-auto-tool-choice --tool-call-parser glm47 --safetensors-load-strategy lazy \
    --enable-cumem-allocator --seed 42 --enable-prompt-tokens-details \
    --enable-force-include-usage --numa-bind \
    --kernel-config '{"enable_flashinfer_autotune":false}'
}
~~~

各主机执行对应命令：

~~~bash
# 81
launch_instance P yyf-glm53flash-c256-p0-81 0,1,2,3 4 18483 19083 19550
launch_instance P yyf-glm53flash-c256-p1-81 4,5,6,7 4 18493 19093 19650
# 82
launch_instance P yyf-glm53flash-c256-p0-82 0,1,2,3 4 18483 19083 19550
launch_instance D yyf-glm53flash-c256-d1-82 4,5,6,7 4 18493 19093 19650
# 83
launch_instance D yyf-glm53flash-c256-d0-83 0,1,2,3,4,5,6,7 8 18483 19083 19550
# 84
launch_instance D yyf-glm53flash-c256-d0-84 0,1,2,3,4,5,6,7 8 18483 19083 19550
~~~

## 4. Router：83

~~~bash
docker run -d --name yyf-glm53flash-c256-router83 --runtime=runc --network host \
  --memory 4g --cpus 8 -e NVIDIA_VISIBLE_DEVICES=void -e OPENAI_API_KEY="$API_KEY" \
  --entrypoint python3 "$ROUTER_IMAGE" -m vllm_router.launch_router \
  --vllm-pd-disaggregation --kv-connector mooncake \
  --prefill http://11.8.0.81:18483 19083 --prefill http://11.8.0.81:18493 19093 \
  --prefill http://11.8.0.82:18483 19083 \
  --decode http://11.8.0.82:18493 --decode http://11.8.0.83:18483 --decode http://11.8.0.84:18483 \
  --prefill-policy cache_aware --decode-policy cache_aware \
  --balance-abs-threshold 4 --balance-rel-threshold 1.5 --intra-node-data-parallel-size 1 \
  --host 11.8.0.83 --port 18300 --prometheus-host 11.8.0.83 --prometheus-port 18301 \
  --disable-retries --worker-startup-timeout-secs 600 --worker-startup-check-interval 5 \
  --log-level info --api-key "$API_KEY"
~~~

## 5. 压测与本机监控

模型容器和部署目录沿用创建时的c256标识；实际正式测试为C288/U432/N432。

max_concurrency=288、max_active_sessions=432，SWE首432条完整自然轨迹，流式usage开启，
reasoning_effort=xhigh、enable_thinking=true，客户端总deadline关闭。
以总计划轮数55%为起点，同成功cohort完整分钟统计，包含命中/未命中输入。

本轮自动编排入口在83；运行中不重复执行：

~~~bash
python3 /data/bench/eng-yangyufei/glm53flash-pd-828384-20261027/integration-20261028/launch_ep4pd.py
~~~

实际命令、server_info、验收与DB路径保存在83：

~~~text
/data/bench/eng-yangyufei/glm53flash-pd-828384-20261027/integration-20261028/ep4p-ep4ep8d-c256-20261029/
~~~

Mac查看逐DP、实例和总速度、running/waiting：

~~~bash
python3 /Users/yyf/workspacke/glm53flash-pd-828384-20261027/tooling/watch_pd.py --ssh
~~~

P取local_compute token差分，不包括GPU前缀命中和外部KV读回；D取generation token差分。
