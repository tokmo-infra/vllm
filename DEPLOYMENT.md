# GLM5.3 Flash 1103 · 部署与配置

推理参数以5595.45万那轮为模板，使用共享分词、FP8 KV、原生MTP3、异步调度及Mooncake缓存。示例展开P81、D83/D84；按角色替换实际服务IP、GPU分组和网卡。

执行前确认目标GPU、端口和共享目录归本次服务使用，宿主保留至少16GiB可用内存。

## 1. 角色与参考参数

| 组件 | 配置 |
| --- | --- |
| P实例 | TP1/DP4/EP4，每DP seq24、batch2048，原生图自动上限192，util0.95；API4、接收线程4 |
| D实例 | TP1/DP8/EP8，每DP seq16、batch1024，FULL_DECODE_ONLY/64，util0.95 |
| 公共 | GLM-5.3 Flash NVFP4，KV FP8，MTP3，prefix256，max-model-len auto |
| Mooncake | 每台模型主机750GiB；水位98%、local_first；Master目标大块60318720字节、64槽、每1000ms检查 |
| Router | 21c9原生Router；P/D均cache_aware，负载阈值4 / 1.5，登记全部实例 |
| Tokenizer | 每机共享池 + 每 API 64 MiB L1，合计 16 GiB/台，等效 2 GiB/GPU |

P端API进程4、KV接收线程4，P/D异步查找开启。FlashInfer在原生预热阶段自动调优。

末段容量预算默认值为0，启用方法见第9节。

### 构建P / D镜像

在仓库根目录执行。增量镜像继承固定基底的原生扩展、FlashInfer及共享Tokenizer，纳入本分支缓存恢复和可选准入代码。

~~~bash
docker build -f integrations/glm53flash/Dockerfile \
  --build-arg BASE_IMAGE=sha256:2c91fd527d05992e2b373938c44556fc94259ce6a6069a7158c2d1f2568cd828 \
  -t local/glm53flash-vllm:1103-p .
docker build -f integrations/glm53flash/Dockerfile \
  --build-arg BASE_IMAGE=sha256:a933ba26983722a147849be4c4e8d13fa4db8e6751e13e05f25ed374f85410f7 \
  -t local/glm53flash-vllm:1103-d1 .
docker build -f integrations/glm53flash/Dockerfile \
  --build-arg BASE_IMAGE=sha256:38a1cea116271a2b886339b4788ddb31f0dcba96a94bf897c734ed9c6df40cb1 \
  -t local/glm53flash-vllm:1103-d2 .
~~~

各主机先设置以下变量，密钥与工作目录按环境填写。

~~~bash
# P/D及Master镜像按本文构建，Store和Router使用固定产物。
export P_IMAGE=local/glm53flash-vllm:1103-p
export D1_IMAGE=local/glm53flash-vllm:1103-d1
export D2_IMAGE=local/glm53flash-vllm:1103-d2
export STORE_IMAGE=sha256:bd6c77d709b8225ebf7b4a9a3890509a4e2df87af846fcc7230b65a7a57d16f7
export MASTER_IMAGE=local/glm53flash-mooncake-master:1103
export ROUTER_IMAGE=sha256:21c9cfa19e348f84a99102ab02922d5cd3a1ef6d659881cac09e9d6479c4d07b
export MODEL_DIR=/data/models/GLM-5.3-Flash-NVFP4
export BASE_WORK=/data/services/glm53flash-1103
export WORK="$BASE_WORK" CACHE_DIR="$BASE_WORK/cache"
export CACHE_NAMESPACE=glm53flash-prefix256
read -rs -p "API key: " API_KEY; echo; export API_KEY
export MASTER_IP=11.8.0.83
export P_IP=11.8.0.81
export D1_IP=11.8.0.83 D2_IP=11.8.0.84
export ROUTER_IP="$D1_IP"
export MC_EVICTION_FIT_REQUEST_BYTES=60318720
export MC_EVICTION_FIT_MIN_SLOTS=64
export MC_EVICTION_FIT_CHECK_MS=1000
mkdir -p "$WORK" "$CACHE_DIR"
# 每台主机执行；P0/P1挂载同一目录。
export TOKENIZER_DIR=/dev/shm/glm53flash-1103/tokenizer
mkdir -p "$TOKENIZER_DIR"
chmod 700 "$TOKENIZER_DIR"
# 冷启动建立epoch，已有epoch原样保留。
if [ ! -e "$TOKENIZER_DIR/epoch" ]; then
  test -z "$(find "$TOKENIZER_DIR" -mindepth 1 -maxdepth 1 -print -quit)" || {
    echo '请先核对现有共享池的代际状态'; exit 1;
  }
  (umask 077; set -o noclobber; printf '%s' "glm1103-$(date +%s%N)" > "$TOKENIZER_DIR/epoch")
fi
test -s "$TOKENIZER_DIR/epoch"
~~~

共享池容量是逻辑预算；请确认宿主 tmpfs 空间、内存和容器权限足够，并预留至少 16 GiB 宿主内存。换轮需要清空缓存时，先停止使用该池的全部服务，再清理池并重建 epoch；勿在运行中直接删除数据库。

| 环境变量 | P分组示例：8 API/台 | D示例：1 API/台 |
| --- | --- | --- |
| VLLM_GLM53_TOKENIZER_CACHE_GIB | 2 | 16 |
| VLLM_GLM53_TOKENIZER_SHARED_GIB | 15.5 | 15.9375 |
| VLLM_GLM53_TOKENIZER_LOCAL_L1_MIB | 64 | 64 |
| 共享池 + L1 | 15.5 + 8 × 0.0625 = 16 GiB | 15.9375 + 0.0625 = 16 GiB |

## 2. Mooncake master

配套Master从固定源码构建，包含实际占用计账和大块空间回收：

~~~bash
git clone https://github.com/kvcache-ai/Mooncake.git /tmp/mooncake-1103-source
git -C /tmp/mooncake-1103-source checkout 719735896c86b56fabec6cf3e825fb2ea640597a
bash integrations/glm53flash/build-mooncake-master.sh \
  /tmp/mooncake-1103-source /tmp/mooncake-1103-build ./mooncake-master-runtime
docker build -f integrations/glm53flash/Dockerfile.mooncake-master \
  -t local/glm53flash-mooncake-master:1103 .
~~~

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

## 3. Mooncake Store：每台模型主机执行一次

NODE_IP取对应主机的服务通信IP。示例使用mlx5_0至mlx5_7、GID index 3；按实际主机网卡填写。SSH管理地址与这些通信地址可以不同。

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

## 5. P角色：按GPU分组启动

示例将一台机器分成两组：P0使用GPU0—3及对应CX8 HCA，P1使用GPU4—7及对应CX8 HCA。按实际GPU与EP配置调整分组。

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
    -e VLLM_EXPERIMENTAL_HOT_TAIL_ADMISSION=0 \
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

## 6. D角色：每个实例执行一次

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

FlashInfer在原生预热阶段自动调优，并复用匹配的持久化结果。

P/D的 block-size CLI均为64，GLM混合缓存最终物理页实际解析为8960。
prefix-match-unit 两端均为256。所有相关节点使用同一Store命名空间，decode保存沿用原生默认false。

## 7. Router

读取各实例实际GPU KV容量，生成P计算量／D请求数与KV容量路由配置。需要curl和jq。下面展开两个P地址和两个D地址作为示例；扩容时同步增加workers配置与对应的--prefill / --decode地址。

~~~bash
export WORK="$BASE_WORK"  # Router使用独立工作目录
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

Router直接连接vLLM原生HTTP监听器。P的MC_ENABLE_DEST_DEVICE_AFFINITY=1提供目的HCA亲和提示，与GPU到本地网卡的绑定共同配置RDMA路径。

启动顺序：Master → 各主机Store → D服务 → P服务 → Router。
模型输出usage由 enable-prompt-tokens-details、enable-force-include-usage 提供；
流式请求传 stream_options.include_usage=true，思考参数沿用 enable_thinking=true / xhigh。

## 8. 指标端口

| 指标 | 端口 / 路径 |
| --- | --- |
| P / D vLLM | 各实例API端口的`/metrics` |
| Router | 18301 /metrics |
| Mooncake master | 61243 /metrics |
| Mooncake Store HTTP | 61245 |

P 速度取 local_compute token计数差，不包括缓存命中；D 速度取generation token计数差。
多路DP按engine标签分别读取，汇总时按实际采样间隔计算。

## 9. 配置项

| 配置 | 默认值 | 本文示例 / 作用 |
| --- | --- | --- |
| Tokenizer共享路径 VLLM_GLM53_TOKENIZER_SHARED_PATH | 空 | /run/glm-tokenizer/segments.db，共享同机精确分词结果。 |
| Tokenizer共享容量 VLLM_GLM53_TOKENIZER_SHARED_GIB | 8 GiB | P为15.5、D为15.9375 GiB。 |
| 私有L1 VLLM_GLM53_TOKENIZER_LOCAL_L1_MIB | 64 MiB/API | 与共享池合计16 GiB/台。 |
| Tokenizer代际文件 VLLM_GLM53_TOKENIZER_CACHE_EPOCH_FILE | 空 | /run/glm-tokenizer/epoch，统一失效各API的L1。 |
| 目的HCA亲和 MC_ENABLE_DEST_DEVICE_AFFINITY | 未设置 | P设为1，给远端网卡选择提供亲和提示。 |
| Master目标大块 MC_EVICTION_FIT_REQUEST_BYTES | 0 | 60318720字节；按大块可分配槽数触发回收。 |
| Master槽数阈值 MC_EVICTION_FIT_MIN_SLOTS | 64 | 64。 |
| Master检查间隔 MC_EVICTION_FIT_CHECK_MS | 1000 ms | 1000。 |
| Router容量配置 VLLM_ROUTER_PD_TOKEN_BALANCE_CONFIG | 空 | /run/token-balance.json，按计算量和KV容量分流。 |
| P末段准入 VLLM_EXPERIMENTAL_HOT_TAIL_ADMISSION | 0 | 精确设为1开启。 |
| P末段事件 VLLM_HOT_TAIL_ADMISSION_DIAGNOSTICS | 0 | 设为1记录有限的登记、延后和消耗首事件。 |

### 可选末段准入预算

异步加载的Mamba请求在prefill末段可能需要额外CoW状态块。该预算在已识别的同末物理块请求间保护这部分余量，并延后竞争请求的准入，减少抢占重算。

默认值为0。需要使用时，在P的docker run中设置VLLM_EXPERIMENTAL_HOT_TAIL_ADMISSION=1；原生async、FP8、MTP和CUDA graph保持原配置。适用范围是首次、纯外部异步加载、同一物理末块的producer请求，应按实际业务验证吞吐与延迟。
