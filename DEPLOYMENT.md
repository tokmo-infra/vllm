# 1P2D · GLM-5.3 Flash 共享 Tokenizer 基线

历史基线：run-8bb8d266fb55，SWE8w C230/U345/N345，55% 后完整窗口 5595.45 万 TPM。

## 1. 拓扑与参数

| 组件 | 配置 |
| --- | --- |
| P | P角色：两个 TP1/DP4/EP4，每DP seq24、batch2048，util0.95 |
| D | D1、D2 各 TP1/DP8/EP8，每DP seq16、batch1024，graph64，util0.95 |
| 公共 | GLM-5.3 Flash NVFP4，KV FP8，MTP3，prefix256，max-model-len auto |
| Mooncake | P、D1、D2 各750 GiB，共2250 GiB；水位98%；local_first |
| Router | 原生 21c9，P/D 均 cache_aware，负载阈值4 / 1.5，无 API 代理、亲和或私有 UDS |
| Tokenizer | 每机共享池 + 每 API 64 MiB L1，合计 16 GiB/台，等效 2 GiB/GPU |

P端API进程4、KV接收线程4，P/D异步查找开启。保留原生MTP3和原生后端，不启用P-only实验，不传禁用FlashInfer自动调优的参数。

各主机先设置以下变量，密钥与工作目录按环境填写。

~~~bash
export IMAGE=local/glm53flash-vllm:0.32.0
export ROUTER_IMAGE=local/glm53flash-router:0.1.16
# 每台机器按角色选择本机已有的历史共享镜像 ID（并非统一 ID）。
# P81: sha256:2c91fd527d05992e2b373938c44556fc94259ce6a6069a7158c2d1f2568cd828
# D83: sha256:a933ba26983722a147849be4c4e8d13fa4db8e6751e13e05f25ed374f85410f7
# D84: sha256:38a1cea116271a2b886339b4788ddb31f0dcba96a94bf897c734ed9c6df40cb1
export MODEL_IMAGE_ID='<本机对应共享镜像完整ID>'
export ROUTER_IMAGE_ID=sha256:21c9cfa19e348f84a99102ab02922d5cd3a1ef6d659881cac09e9d6479c4d07b
docker tag "$MODEL_IMAGE_ID" "$IMAGE"
docker tag "$ROUTER_IMAGE_ID" "$ROUTER_IMAGE"
export MODEL_DIR=/data/models/GLM-5.3-Flash-NVFP4
export WORK=/path/to/glm-pd
export CACHE_DIR="$WORK/cache"
export CACHE_NAMESPACE=glm53flash-prefix256-v1
read -rs -p "API key: " API_KEY; echo; export API_KEY
export MASTER_IP=11.8.0.83
export P_IP=11.8.0.81
export D1_IP=11.8.0.83 D2_IP=11.8.0.84
export ROUTER_IP="$D1_IP"
mkdir -p "$WORK" "$CACHE_DIR"
# 每台主机单独初始化；P0/P1 必须挂载同一目录。仅在服务停止且需要清空时删除旧池。
export TOKENIZER_DIR=/dev/shm/glm53-tokenizer
mkdir -p "$TOKENIZER_DIR"
chmod 700 "$TOKENIZER_DIR"
# 仅在所有使用该池的服务停止时初始化或更新 epoch。
printf '%s' 'admission-generation' > "$TOKENIZER_DIR/epoch"
~~~

共享池容量是逻辑预算；请确认宿主 tmpfs 空间、内存和容器权限足够，并预留至少 16 GiB 宿主内存。换轮需要清空缓存时，先停止使用该池的全部服务，再清理池并重建 epoch；勿在运行中直接删除数据库。

| 环境变量 | P81：8 API/台 | D83/84：1 API/台 |
| --- | --- | --- |
| VLLM_GLM53_TOKENIZER_CACHE_GIB | 2 | 16 |
| VLLM_GLM53_TOKENIZER_SHARED_GIB | 15.5 | 15.9375 |
| VLLM_GLM53_TOKENIZER_LOCAL_L1_MIB | 64 | 64 |
| 共享池 + L1 | 15.5 + 8 × 0.0625 = 16 GiB | 15.9375 + 0.0625 = 16 GiB |

## 2. Mooncake master

~~~bash
docker run -d --name glm-mooncake-master --network host --memory 4g \
  --runtime=runc -e NVIDIA_VISIBLE_DEVICES=void \
  -e MC_TCP_BIND_ADDRESS="$MASTER_IP" -e MOONCAKE_REQUESTER_LOCAL_HOSTNAME="$MASTER_IP" \
  --entrypoint mooncake_master "$IMAGE" \
  --rpc_port=61241 --enable_http_metadata_server=true \
  --http_metadata_server_host=0.0.0.0 --http_metadata_server_port=61242 \
  --metrics_port=61243 --eviction_high_watermark_ratio=0.98 \
  --allocation_strategy=local_first --logtostderr=true
~~~

## 3. Mooncake Store：P、D1、D2 各执行一次

NODE_IP 取对应主机RDMA IP。三台Store均使用 mlx5_0 至 mlx5_7；GID index为3。

~~~bash
export LIBCUDA_PATH=/path/to/host/libcuda.so.1
export NODE_IP="$D1_IP"  # P用P_IP；D2用D2_IP
export HCAS=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
docker run -d --name glm-mooncake-store --network host --ipc host \
  --memory 832g --memory-swap 832g \
  --runtime=runc -e NVIDIA_VISIBLE_DEVICES=void --ulimit memlock=-1:-1 \
  -e MC_GID_INDEX=3 -e MC_TCP_BIND_ADDRESS="$NODE_IP" \
  -e MOONCAKE_REQUESTER_LOCAL_HOSTNAME="$NODE_IP" \
  --cap-add IPC_LOCK --cap-add SYS_NICE \
  -v "$LIBCUDA_PATH:/opt/host-driver/libcuda.so.1:ro" \
  -e LD_LIBRARY_PATH=/opt/host-driver:/usr/local/cuda/lib64 \
  --device /dev/infiniband --entrypoint mooncake_client "$IMAGE" \
  --master_server_address="$MASTER_IP:61241" \
  --metadata_server="http://$MASTER_IP:61242/metadata" \
  --protocol=rdma --host="$NODE_IP" --port=61244 \
  --device_names="$HCAS" --global_segment_size=805306368000 \
  --local_buffer_size=0 --enable_http_server=true --http_port=61245 \
  --enable_offload=false --logtostderr=true
~~~

750 GiB为每台主机共享池，不是每张GPU各分配750 GiB。宿主至少保留16 GiB可用内存。

## 4. 模型客户端 Mooncake 配置

在每台模型主机定义以下函数。P0/P1分别生成独立配置，D各生成一份配置。
客户端不重复创建共享池，仅注册2 GiB本地传输缓冲。

~~~bash
write_mc_config() {
  cat > "$1" <<EOF
{
  "mode": "standalone-store",
  "metadata_server": "http://$MASTER_IP:61242/metadata",
  "master_server_address": "$MASTER_IP:61241",
  "protocol": "rdma",
  "device_name": "$2",
  "global_segment_size": 0,
  "local_buffer_size": 2147483648,
  "enable_offload": false,
  "tenant_id": "default"
}
EOF
}
~~~

## 5. P角色：两个 TP1/DP4/EP4 实例

P0使用GPU0—3及对应CX8 HCA，P1使用GPU4—7及对应CX8 HCA。

~~~bash
export NET_IF=enp115s0f1np1
for IDX in 0 1; do
  if [ "$IDX" = 0 ]; then
    GPU_REQUEST='"device=0,1,2,3"'
    HCAS=mlx5_0,mlx5_1,mlx5_2,mlx5_3
    IB_HCA='=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1'
    PORT=18483; BOOTSTRAP=19083; RPC=19550
  else
    GPU_REQUEST='"device=4,5,6,7"'
    HCAS=mlx5_4,mlx5_5,mlx5_6,mlx5_7
    IB_HCA='=mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1'
    PORT=18493; BOOTSTRAP=19093; RPC=19650
  fi
  write_mc_config "$WORK/p$IDX-mooncake.json" "$HCAS"
  ENGINE_ID="glm-P$IDX-$(date +%s%N)"
  P_KV=$(cat <<EOF
{
  "kv_connector":"MultiConnector","kv_role":"kv_both","engine_id":"$ENGINE_ID",
  "kv_load_failure_policy":"fail",
  "kv_connector_extra_config":{"connectors":[
    {"kv_connector":"MooncakeConnector","kv_role":"kv_producer",
     "kv_connector_extra_config":{"mooncake_protocol":"rdma","device_name":"$HCAS"}},
    {"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both",
     "kv_connector_extra_config":{"cache_prefix":"$CACHE_NAMESPACE",
       "load_async":true,"lookup_async":true,"store_compact_group_payloads":true}}
  ]}
}
EOF
)
  docker run -d --name "glm-P$IDX" --init --network host --shm-size 64g \
    --memory 96g --memory-swap 96g --gpus "$GPU_REQUEST" \
    --device /dev/infiniband --cap-add IPC_LOCK --cap-add SYS_NICE \
    --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 \
    -v "$MODEL_DIR:/model:ro" -v "$WORK:/work" -v "$CACHE_DIR/p$IDX:/root/.cache" \
    -v "$TOKENIZER_DIR:/run/glm-tokenizer:rw" \
    -e VLLM_GLM53_TOKENIZER_SHARED_PATH=/run/glm-tokenizer/segments.db \
    -e VLLM_GLM53_TOKENIZER_SHARED_GIB=15.5 -e VLLM_GLM53_TOKENIZER_CACHE_GIB=2 \
    -e VLLM_GLM53_TOKENIZER_LOCAL_L1_MIB=64 \
    -e VLLM_GLM53_TOKENIZER_CACHE_EPOCH_FILE=/run/glm-tokenizer/epoch \
    -e MC_TCP_BIND_ADDRESS="$P_IP" -e MOONCAKE_REQUESTER_LOCAL_HOSTNAME="$P_IP" \
    -e MC_ENABLE_DEST_DEVICE_AFFINITY=1 \
    -e VLLM_HOST_IP="$P_IP" -e NCCL_SOCKET_IFNAME="$NET_IF" \
    -e GLOO_SOCKET_IFNAME="$NET_IF" -e NCCL_IB_HCA="$IB_HCA" \
    -e NCCL_IB_GID_INDEX=3 -e MC_GID_INDEX=3 -e NCCL_DEBUG=INFO \
    -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_SERVER_DEV_MODE=1 -e VLLM_MOONCAKE_LOAD_RECV_THREADS=4 \
    -e MOONCAKE_CONFIG_PATH="/work/p$IDX-mooncake.json" \
    -e VLLM_MOONCAKE_BOOTSTRAP_PORT="$BOOTSTRAP" -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120 \
    --entrypoint vllm "$IMAGE" serve /model \
    --served-model-name glm-5.3-flash-nvfp4 --host "$P_IP" --port "$PORT" \
    --api-key "$API_KEY" --api-server-count 4 --language-model-only \
    --tensor-parallel-size 1 --data-parallel-size 4 --data-parallel-size-local 4 \
    --data-parallel-backend mp --data-parallel-rpc-port "$RPC" \
    --enable-expert-parallel --enable-ep-weight-filter \
    --max-model-len auto --max-num-seqs 24 --max-num-batched-tokens 2048 \
    --gpu-memory-utilization 0.95 --kv-cache-dtype fp8 \
    --block-size 64 --prefix-match-unit 256 --mamba-cache-mode align \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --kv-transfer-config "$P_KV" --scheduler-reserve-full-isl \
    --enable-prefix-caching --reasoning-parser glm47 \
    --enable-auto-tool-choice --tool-call-parser glm47 \
    --safetensors-load-strategy lazy --enable-cumem-allocator --seed 42 \
    --enable-prompt-tokens-details --enable-force-include-usage --numa-bind
done
~~~

## 6. D1、D2各执行一次

NODE_IP、ROLE_ID、NET_IF 按当前D角色填写。

~~~bash
export HCAS=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
write_mc_config "$WORK/mooncake.json" "$HCAS"
export ROLE_ID=D1  # 另一D角色使用D2
export NODE_IP="$D1_IP"
export NET_IF=enp115s0f1np1
export IB_HCA='=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1'
ENGINE_ID="glm-D-$ROLE_ID-$(date +%s%N)"
D_KV=$(cat <<EOF
{
  "kv_connector":"MultiConnector","kv_role":"kv_both","engine_id":"$ENGINE_ID",
  "kv_load_failure_policy":"fail",
  "kv_connector_extra_config":{"connectors":[
    {"kv_connector":"MooncakeConnector","kv_role":"kv_consumer",
     "kv_connector_extra_config":{"mooncake_protocol":"rdma",
       "device_name":"$HCAS"}},
    {"kv_connector":"MooncakeStoreConnector","kv_role":"kv_consumer",
     "kv_connector_extra_config":{"cache_prefix":"$CACHE_NAMESPACE",
       "load_async":true,"lookup_async":true,"store_compact_group_payloads":true}}
  ]}
}
EOF
)
docker run -d --name glm-D --init --memory 96g --memory-swap 96g --network host --shm-size 64g \
  --cap-add IPC_LOCK --cap-add SYS_NICE --ulimit stack=67108864:67108864 \
  --gpus all --device /dev/infiniband --ulimit memlock=-1:-1 \
  -v "$MODEL_DIR:/model:ro" -v "$WORK:/work" -v "$CACHE_DIR:/root/.cache" \
  -v "$TOKENIZER_DIR:/run/glm-tokenizer:rw" \
  -e VLLM_GLM53_TOKENIZER_SHARED_PATH=/run/glm-tokenizer/segments.db \
  -e VLLM_GLM53_TOKENIZER_SHARED_GIB=15.9375 -e VLLM_GLM53_TOKENIZER_CACHE_GIB=16 \
  -e VLLM_GLM53_TOKENIZER_LOCAL_L1_MIB=64 \
  -e VLLM_GLM53_TOKENIZER_CACHE_EPOCH_FILE=/run/glm-tokenizer/epoch \
  -e MC_TCP_BIND_ADDRESS="$NODE_IP" -e MOONCAKE_REQUESTER_LOCAL_HOSTNAME="$NODE_IP" \
  -e VLLM_HOST_IP="$NODE_IP" -e NCCL_SOCKET_IFNAME="$NET_IF" \
  -e GLOO_SOCKET_IFNAME="$NET_IF" -e NCCL_IB_HCA="$IB_HCA" \
  -e NCCL_IB_GID_INDEX=3 -e MC_GID_INDEX=3 -e NCCL_DEBUG=INFO \
  -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_SERVER_DEV_MODE=1 \
  -e MOONCAKE_CONFIG_PATH=/work/mooncake.json \
  -e VLLM_MOONCAKE_BOOTSTRAP_PORT=19083 -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120 \
  --entrypoint vllm "$IMAGE" serve /model \
  --served-model-name glm-5.3-flash-nvfp4 --host "$NODE_IP" --port 18483 \
  --api-key "$API_KEY" --api-server-count 1 --language-model-only \
  --tensor-parallel-size 1 --data-parallel-size 8 --data-parallel-size-local 8 \
  --data-parallel-backend mp --data-parallel-rpc-port 19550 \
  --enable-expert-parallel --enable-ep-weight-filter \
  --max-model-len auto --max-num-seqs 16 --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.95 --kv-cache-dtype fp8 \
  --block-size 64 --prefix-match-unit 256 --mamba-cache-mode align \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --kv-transfer-config "$D_KV" --scheduler-reserve-full-isl \
  --enable-prefix-caching --reasoning-parser glm47 \
  --enable-auto-tool-choice --tool-call-parser glm47 \
  --safetensors-load-strategy lazy --enable-cumem-allocator --seed 42 \
  --enable-prompt-tokens-details --enable-force-include-usage --numa-bind \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":64}'
~~~

未传kernel-config禁用项，FlashInfer在原生预热阶段自动调优，并复用匹配的持久化结果。

P/D的 block-size CLI均为64，GLM混合缓存最终物理页实际解析为8960。
prefix-match-unit 两端均为256。三个节点使用同一Store命名空间，decode保存沿用原生默认false。

## 7. Router

读取当前实例实际GPU KV容量，生成P计算量／D请求数与KV容量路由配置。需要curl和jq。

~~~bash
kv_tokens() {
  curl -fsS -H "Authorization: Bearer $API_KEY" "$1/server_info?config_format=json" |
    jq -r '.vllm_config.cache_config.kv_cache_size_tokens'
}
P0_KV=$(kv_tokens "http://$P_IP:18483")
P1_KV=$(kv_tokens "http://$P_IP:18493")
D1_KV=$(kv_tokens "http://$D1_IP:18483")
D2_KV=$(kv_tokens "http://$D2_IP:18483")
cat > "$WORK/token-balance.json" <<EOF
{
  "tokenizer_path":"/model/tokenizer.json",
  "kv_fraction":0.85,
  "workers":{
    "http://$P_IP:18483":{"kv_tokens":$P0_KV,"max_seqs":24},
    "http://$P_IP:18493":{"kv_tokens":$P1_KV,"max_seqs":24},
    "http://$D1_IP:18483":{"kv_tokens":$D1_KV,"max_seqs":16},
    "http://$D2_IP:18483":{"kv_tokens":$D2_KV,"max_seqs":16}
  }
}
EOF
docker run -d --name glm-router --runtime=runc --network host \
  --memory 8g --memory-swap 8g \
  -v "$MODEL_DIR:/model:ro" -v "$WORK/token-balance.json:/run/token-balance.json:ro" \
  -e VLLM_ROUTER_PD_TOKEN_BALANCE_CONFIG=/run/token-balance.json \
  -e NVIDIA_VISIBLE_DEVICES=void -e OPENAI_API_KEY="$API_KEY" \
  -e MC_TCP_BIND_ADDRESS="$ROUTER_IP" -e MOONCAKE_REQUESTER_LOCAL_HOSTNAME="$ROUTER_IP" \
  --entrypoint /usr/local/bin/vllm-router "$ROUTER_IMAGE" \
  --vllm-pd-disaggregation --kv-connector mooncake \
  --prefill "http://$P_IP:18483" 19083 --prefill "http://$P_IP:18493" 19093 \
  --decode "http://$D1_IP:18483" --decode "http://$D2_IP:18483" \
  --prefill-policy cache_aware --decode-policy cache_aware \
  --balance-abs-threshold 4 --balance-rel-threshold 1.5 \
  --intra-node-data-parallel-size 1 --host "$ROUTER_IP" --port 18300 \
  --prometheus-host "$ROUTER_IP" --prometheus-port 18301 \
  --disable-retries --worker-startup-timeout-secs 600 \
  --worker-startup-check-interval 5 --log-level info --api-key "$API_KEY"
~~~

Router 直接连接 vLLM 原生 HTTP 监听器，不部署 API 代理、亲和或私有 UDS。

启动顺序：master → 三台Store → 两个D → 两个P → Router。
模型输出usage由 enable-prompt-tokens-details、enable-force-include-usage 提供；
流式请求传 stream_options.include_usage=true，思考参数沿用 enable_thinking=true / xhigh。
本分支命令不包含临时计时RPC、调试挂载或自动压测。

## 8. 指标端口

| 指标 | 端口 / 路径 |
| --- | --- |
| P / D vLLM | 18483 /metrics |
| Router | 18301 /metrics |
| Mooncake master | 61243 /metrics |
| Mooncake Store HTTP | 61245 |

P 速度取 local_compute token计数差，不包括缓存命中；D 速度取generation token计数差。
多路DP按engine标签分别读取，汇总时按实际采样间隔计算。
