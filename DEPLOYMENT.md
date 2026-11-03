# 1P2D · 当前共享 Tokenizer + HCA + PUT/回退配置

2026-11-03只读核对当前容器：85为P（双EP4），86/87各独立EP8 D，Master/Router在86，无压测客户端。以下是**冷启动复现命令**，不是要求重启现有服务；不得在在用机器重复启动或清空Tokenizer epoch/共享池。

沿用历史 `run-8bb8d266fb55`（5595.45万TPM）的原始数据和推理参数，只增加P目的HCA亲和、PUT压力恢复和Master计账/大块压力淘汰。新轮 `run-af1416ddcbde` 为5421.869万TPM、TTFT6.120s、0请求失败。比同机基线5429.736万吞吐-0.145%、TTFT-3.96%；保留理由是缓存可靠性与延迟，不宣称吞吐提升或HTTP500普适根治。

## 1. 拓扑与参数

| 组件 | 配置 |
| --- | --- |
| P85 | 两个TP1/DP4/EP4，每DP seq24、batch2048，原生图自动上限192，util0.95 |
| D86/87 | 各TP1/DP8/EP8，每DP seq16、batch1024，FULL_DECODE_ONLY/64，util0.95 |
| 公共 | GLM-5.3 Flash NVFP4，KV FP8，MTP3，prefix256，max-model-len auto |
| Mooncake | 三台各750GiB，共2250GiB；水位98%、local_first；Master fit=60318720 bytes/64 slots/1000ms |
| Router | 原生21c9，P/D均cache_aware，负载阈值4 / 1.5，无API代理、API前缀亲和或私有UDS |
| Tokenizer | 每机共享池 + 每 API 64 MiB L1，合计 16 GiB/台，等效 2 GiB/GPU |

P端API进程4、KV接收线程4，P/D异步查找开启。保留原生MTP3和原生后端，不启用P-only实验，不传禁用FlashInfer自动调优的参数。

抢占预算实验仅保留为默认关闭的 `VLLM_EXPERIMENTAL_HOT_TAIL_ADMISSION` 环境变量；以下基线命令不启用它。只在包含新源码的镜像中对 P 显式设为 `1` 才会尝试启用，旧镜像不会因设置变量自动获得功能。范围、退出条件和两轮无吞吐收益的结果见 [实验说明](docs/design/hot_tail_admission.md)。

各主机先设置以下变量，密钥与工作目录按环境填写。

~~~bash
# 使用已验证镜像，不覆盖历史标签，不将repo HEAD整体覆盖旧镜像。
export P_IMAGE=sha256:643e95964a8c12483394d4689f0d78a84bb8319a09f18084cbfbc2191c4c1bca
export D1_IMAGE=sha256:a933ba26983722a147849be4c4e8d13fa4db8e6751e13e05f25ed374f85410f7
export D2_IMAGE=sha256:38a1cea116271a2b886339b4788ddb31f0dcba96a94bf897c734ed9c6df40cb1
export STORE_IMAGE=sha256:bd6c77d709b8225ebf7b4a9a3890509a4e2df87af846fcc7230b65a7a57d16f7
export MASTER_IMAGE=sha256:8c7b3faf37167405712c29da0cef1cc330f62a704db5ebd11f175d6646a6a27a
export ROUTER_IMAGE=sha256:21c9cfa19e348f84a99102ab02922d5cd3a1ef6d659881cac09e9d6479c4d07b
export MODEL_DIR=/data/models/GLM-5.3-Flash-NVFP4
export BASE_WORK=/home/eng-yangyufei/yyf/glm-http500-shared-20261102
export WORK="$BASE_WORK" CACHE_DIR="$BASE_WORK/cache"
export CACHE_NAMESPACE=glm-http500-shared-20261102-prefix256
read -rs -p "API key: " API_KEY; echo; export API_KEY
export MASTER_IP=11.8.0.86
export P_IP=11.8.0.85
export D1_IP=11.8.0.86 D2_IP=11.8.0.87
export ROUTER_IP="$D1_IP"
export MC_EVICTION_FIT_REQUEST_BYTES=60318720
export MC_EVICTION_FIT_MIN_SLOTS=64
export MC_EVICTION_FIT_CHECK_MS=1000
mkdir -p "$WORK" "$CACHE_DIR"
# 每台主机单独初始化；P0/P1 必须挂载同一目录。仅在服务停止且需要清空时删除旧池。
export TOKENIZER_DIR=/dev/shm/glm-http500-shared-20261102/tokenizer
mkdir -p "$TOKENIZER_DIR"
chmod 700 "$TOKENIZER_DIR"
# 现有epoch原样保留。仅全新冷启动且无使用者时，才人工初始化：
# printf '%s' 'admission-generation' > "$TOKENIZER_DIR/epoch"
test -s "$TOKENIZER_DIR/epoch" || { echo 'epoch缺失：停止启动并确认冷启动条件'; exit 1; }
~~~

共享池容量是逻辑预算；请确认宿主 tmpfs 空间、内存和容器权限足够，并预留至少 16 GiB 宿主内存。换轮需要清空缓存时，先停止使用该池的全部服务，再清理池并重建 epoch；勿在运行中直接删除数据库。

| 环境变量 | P85：8 API/台 | D86/87：1 API/台 |
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
  -e MC_EVICTION_FIT_REQUEST_BYTES="${MC_EVICTION_FIT_REQUEST_BYTES:?}" \
  -e MC_EVICTION_FIT_MIN_SLOTS="${MC_EVICTION_FIT_MIN_SLOTS:-64}" \
  -e MC_EVICTION_FIT_CHECK_MS="${MC_EVICTION_FIT_CHECK_MS:-1000}" \
  --entrypoint mooncake_master "$MASTER_IMAGE" \
  --rpc_port=61241 --enable_http_metadata_server=true \
  --http_metadata_server_host=0.0.0.0 --http_metadata_server_port=61242 \
  --metrics_port=61243 --eviction_high_watermark_ratio=0.98 \
  --allocation_strategy=local_first --logtostderr=true
~~~

当前Master名称为 `local/glm53flash-mooncake-master:fitpressure-c230-20261103`，P名称为 `local/glm53flash-vllm:put-recovery-c230-20261103`，固定ID如上。Master由Mooncake `719735896c86b56fabec6cf3e825fb2ea640597a` 加 `43cb7086` / `5e9ff92b` 重编译；Store SDK不变，不应声称与其他轮ELF逐字节相同。

仅缺少当前镜像时，可在独立目录重建Master（不覆盖在用镜像/容器）：

~~~bash
git clone https://github.com/kvcache-ai/Mooncake.git /tmp/mooncake-occupied-source
git -C /tmp/mooncake-occupied-source checkout 719735896c86b56fabec6cf3e825fb2ea640597a
bash integrations/glm53flash/build-mooncake-master.sh \
  /tmp/mooncake-occupied-source /tmp/mooncake-occupied-build ./mooncake-master-runtime
docker build -f integrations/glm53flash/Dockerfile.mooncake-master \
  --build-arg BASE_IMAGE=local/glm53flash-vllm:0.32.0 \
  -t local/glm53flash-mooncake-master:fitpressure-20261102 .
# 新构建使用独立标签，验证后才显式改MASTER_IMAGE，不假定与已测ID相同。
# export MASTER_IMAGE=local/glm53flash-mooncake-master:fitpressure-20261102
export MC_EVICTION_FIT_REQUEST_BYTES=60318720
export MC_EVICTION_FIT_MIN_SLOTS=64
export MC_EVICTION_FIT_CHECK_MS=1000
~~~

## 3. Mooncake Store：P、D1、D2 各执行一次

NODE_IP 取对应主机RDMA IP。三台Store均使用 mlx5_0 至 mlx5_7；GID index为3。

~~~bash
export LIBCUDA_PATH=$(readlink -f /lib/x86_64-linux-gnu/libcuda.so.1)
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
  --device /dev/infiniband --entrypoint mooncake_client "$STORE_IMAGE" \
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
export NET_IF=ens19np0
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
  ROLE_WORK="$BASE_WORK/p$IDX"
  mkdir -p "$ROLE_WORK/cache"
  write_mc_config "$ROLE_WORK/mooncake.json" "$HCAS"
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
    -v "$MODEL_DIR:/model:ro" -v "$ROLE_WORK:/work" -v "$ROLE_WORK/cache:/root/.cache" \
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
    -e MOONCAKE_CONFIG_PATH=/work/mooncake.json \
    -e VLLM_MOONCAKE_BOOTSTRAP_PORT="$BOOTSTRAP" -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120 \
    --entrypoint vllm "$P_IMAGE" serve /model \
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
export WORK="$BASE_WORK/d0" CACHE_DIR="$BASE_WORK/d0/cache"
mkdir -p "$CACHE_DIR"
write_mc_config "$WORK/mooncake.json" "$HCAS"
export ROLE_ID=D1  # 另一D角色使用D2
export NODE_IP="$D1_IP" D_IMAGE="$D1_IMAGE"  # D2改为D2_IP / D2_IMAGE
export NET_IF=ens19np0
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
  --entrypoint vllm "$D_IMAGE" serve /model \
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
export WORK="$BASE_WORK"  # Router宿主为86，避免沿用D角色子目录
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

Router直接连接vLLM原生HTTP监听器，不部署API代理、API前缀亲和或私有UDS。P的 `MC_ENABLE_DEST_DEVICE_AFFINITY=1` 是Mooncake目的HCA路由hint，不是API亲和，也不替代原有GPU→本地NIC绑定。

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

## 9. 原始C230压测口径

- 原始SWE8w文件 `swe8w-historical-5595-c345.jsonl`：345会话、25512轮。SHA-256为 `f6fb8de6deee44d727affbc6014b61e7064b0719facfb13fab172305b0475129`。
- C230/U345/N345、自然max4096，不裁片或更换轨迹。完成14032轮达55%，只统计其后完整分钟；在途不足230持续60秒按原规则结束。
- 正式前清本任务GPU/Mooncake/Tokenizer，仅通过已验收的排空/epoch流程，不能直接删除在用文件。
- 当前客户端8CPU/8GiB；它与模型容器96GiB、Tokenizer合计16GiB不是同一预算。
- 不开hot-tail、水位优化、API代理、DeepEP、GET拆批、Owner、自定义失败重算或延长TTL。

## 10. 相对5595.45万历史轮的实际差异

| 项目 | 历史 `run-8bb8d266fb55` | 当前 `run-af1416ddcbde` |
| --- | --- | --- |
| 主机 | P81 / D83、84 | P85 / D86、87；IP/网口随宿主调整 |
| P镜像 | 共享无代理 `2c91fd527d05` | 同基底仅替换PUT worker的 `643e95964a8c`，提交 `7a26c0a8` |
| 目的HCA亲和 | 未显式开启 | 仅两P `MC_ENABLE_DEST_DEVICE_AFFINITY=1`，配置记录 `5706f4e4` |
| Master | 原生Master | `8c7b3faf3716`：`43cb7086`计账 + `5e9ff92b` fit压力，60318720/64/1000 |
| D、Store SDK、Router | 历史共享D、原SDK、21c9 Router | 对应镜像不变，SDK未换，无API代理 |
| 客户端 | 历史native runtime、20GiB限制 | B300同步native runtime、8CPU/8GiB；路径/导出/临时目录设置不同 |
| 路径/身份 | 历史任务目录、namespace、engine ID | 本任务隔离目录/namespace、新engine ID，不是推理调参 |
| 数据/推理 | 原SHA、C230/U345/N345、max4096 | 相同；P24/2048/自动192、D16/1024/图64、util.95均保持 |
| 共享Tokenizer | 2GiB/GPU等效 + 64MiB/API L1 | 相同，整机总预算16GiB，未撤共享 |
| async/FP8/MTP3/fullISL | 原生路径 | 相同；默认通信后端、750GiB/台、98%水位与TTL不改 |
| 自动GPU KV块数 | 每P实例720、每D实例5056 | 每P实例724、每D实例5064；未因补丁缩小KV池 |

### 历史吞吐差距如何拆开

| 同口径11分钟窗口 | 历史共享无API | 同机HCA基线 | 当前HCA+PUT/Master |
| --- | ---: | ---: | ---: |
| 总TPM/万 | 5595.450 | 5429.736 | 5421.869 |
| 窗口成功请求 | 6951 | 6792 | 6773 |
| 单请求平均输入token | 87787.16 | 87178.82 | 87294.42 |
| 单请求平均输出token | 761.18 | 758.61 | 761.93 |
| TTFT均值/s | 6.056 | 6.372 | 6.120 |
| TPOT均值/ms | 20.111 | 20.371 | 20.487 |
| 全程P抢占 | 29 | 18 | 15 |

5595.45万是原native DB的11分钟窗口token总量除以11，并非另换了更短样本或报告卡速。相同数据不保证各次窗口完成完全相同的会话轮次：当前比历史少178条窗口请求，平均输入也稍低；这是吞吐差的数值分解，不是物理根因。

历史→同机HCA基线已下降2.962%，其后增加PUT/Master只再下降0.145%；不能把前面的下降都归给这次PUT改动，也不能把未定位的约3%直接当误差。换机、客户端runtime/20→8GiB等尚未做严格单变量验证；没有证据证明哪一项主导。原始推理参数、异步和共享Tokenizer实际均保留，GPU KV池反而略大；抢占更少也没有自动换来更高TPM。

本次PUT失败键175→0，TTFT P99 13.613→12.813s，额外相邻回退token代理量在5936个匹配窗口对中48128→5376；这是组合效果，不是逐请求PUT因果证明。正式轮没有 `-200`，worker恢复分支仍依赖隔离真实TCP验收；原生HTTP失败两轮均0。
