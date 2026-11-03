# GLM-5.3 Flash · 部署与配置

本文统一说明P / D服务、Mooncake、Router、共享分词和源码构建，不绑定某个并发数或固定P/D数量。按实际机器扩展实例，并在Router中登记全部服务地址。

以下命令用一组IP和GPU分组展开示例。增加实例时分别分配容器名、GPU、端口、工作目录和新的engine ID；跨机EP还需配置对应的rank与通信参数，不能只增加Router地址。冷启动前确认资源未被其他服务使用，保留在用Tokenizer共享池及epoch。

## 1. 角色与参考参数

| 组件 | 配置 |
| --- | --- |
| P实例 | TP1/DP4/EP4，每DP seq24、batch2048，原生图自动上限192，util0.95；API4、接收线程4 |
| D实例 | TP1/DP8/EP8，每DP seq16、batch1024，FULL_DECODE_ONLY/64，util0.95 |
| 公共 | GLM-5.3 Flash NVFP4，KV FP8，MTP3，prefix256，max-model-len auto |
| Mooncake | 每台模型主机750GiB；水位98%、local_first；Master目标大块60318720字节、64槽、每1000ms检查 |
| Router | 分支新构建，聊天请求默认复用P的分词结果；P/D均cache_aware，负载阈值4 / 1.5，无API代理 |
| Tokenizer | 每机共享池 + 每 API 64 MiB L1，合计 16 GiB/台，等效 2 GiB/GPU |

P端API进程4、KV接收线程4，P/D异步查找开启。保留原生MTP3和原生后端，不启用P-only实验，不传禁用FlashInfer自动调优的参数。

末段容量预算默认关闭；可选条件和配置集中在本文第9节。

各主机先设置以下变量，密钥与工作目录按环境填写。

~~~bash
# 示例固定镜像ID；使用自己的构建时，替换为已验证的对应镜像。
export P_IMAGE=sha256:643e95964a8c12483394d4689f0d78a84bb8319a09f18084cbfbc2191c4c1bca
export D1_IMAGE=sha256:a933ba26983722a147849be4c4e8d13fa4db8e6751e13e05f25ed374f85410f7
export D2_IMAGE=sha256:38a1cea116271a2b886339b4788ddb31f0dcba96a94bf897c734ed9c6df40cb1
export STORE_IMAGE=sha256:bd6c77d709b8225ebf7b4a9a3890509a4e2df87af846fcc7230b65a7a57d16f7
export MASTER_IMAGE=sha256:8c7b3faf37167405712c29da0cef1cc330f62a704db5ebd11f175d6646a6a27a
export ROUTER_IMAGE=sha256:c07f71ef603b3c28c2f2b48e0657c4c96fcee285428dcd64747ed53a3d3aeec5
export MODEL_DIR=/data/models/GLM-5.3-Flash-NVFP4
export BASE_WORK=/data/services/glm53flash
export WORK="$BASE_WORK" CACHE_DIR="$BASE_WORK/cache"
export CACHE_NAMESPACE=glm53flash-prefix256
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
export TOKENIZER_DIR=/dev/shm/glm53flash/tokenizer
mkdir -p "$TOKENIZER_DIR"
chmod 700 "$TOKENIZER_DIR"
# 现有epoch原样保留。仅全新冷启动且无使用者时，才人工初始化：
# printf '%s' 'admission-generation' > "$TOKENIZER_DIR/epoch"
test -s "$TOKENIZER_DIR/epoch" || { echo 'epoch缺失：停止启动并确认冷启动条件'; exit 1; }
~~~

共享池容量是逻辑预算；请确认宿主 tmpfs 空间、内存和容器权限足够，并预留至少 16 GiB 宿主内存。换轮需要清空缓存时，先停止使用该池的全部服务，再清理池并重建 epoch；勿在运行中直接删除数据库。

| 环境变量 | P分组示例：8 API/台 | D示例：1 API/台 |
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

当前Master名称为 `local/glm53flash-mooncake-master:fitpressure-c230-20261103`，P名称为 `local/glm53flash-vllm:put-recovery-c230-20261103`，固定ID如上。Master由Mooncake `719735896c86b56fabec6cf3e825fb2ea640597a` 加 `95f4f832` / `bdc687f4` 对应补丁编译；Store SDK不变。重新构建时记录新的ELF和镜像哈希。

需要重建Master时，在独立目录执行：

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

未传kernel-config禁用项，FlashInfer在原生预热阶段自动调优，并复用匹配的持久化结果。

P/D的 block-size CLI均为64，GLM混合缓存最终物理页实际解析为8960。
prefix-match-unit 两端均为256。所有相关节点使用同一Store命名空间，decode保存沿用原生默认false。

## 7. Router

读取各实例实际GPU KV容量，生成P计算量／D请求数与KV容量路由配置。需要curl和jq。下面展开两个P地址和两个D地址作为示例；扩容时同步增加workers配置与对应的--prefill / --decode地址。

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

新Router来自GLM分支的`08d71951`，镜像名称为`local/glm53flash-router:native-ids-20261103`。聊天请求默认向P取得prompt token IDs，再交给D原生复用，减少重复分词；无需额外环境变量。P/D的模型、Tokenizer和模板须一致；不兼容时在Router容器显式设置`VLLM_ROUTER_PD_REUSE_PROMPT_IDS=0`。此路径等P完成后派发D，不关闭vLLM的原生异步调度。

启动顺序：Master → 各主机Store → D服务 → P服务 → Router。
模型输出usage由 enable-prompt-tokens-details、enable-force-include-usage 提供；
流式请求传 stream_options.include_usage=true，思考参数沿用 enable_thinking=true / xhigh。
本分支命令不包含临时计时RPC、调试挂载或自动压测。

## 8. 指标端口

| 指标 | 端口 / 路径 |
| --- | --- |
| P / D vLLM | 各实例API端口的`/metrics` |
| Router | 18301 /metrics |
| Mooncake master | 61243 /metrics |
| Mooncake Store HTTP | 61245 |

P 速度取 local_compute token计数差，不包括缓存命中；D 速度取generation token计数差。
多路DP按engine标签分别读取，汇总时按实际采样间隔计算。

## 9. 开关与可选功能

下表只列实际存在的配置项；没有独立开关的功能随对应代码或镜像生效。

| 配置项 | 默认值 | 示例配置 |
| --- | --- | --- |
| 预调优只加载 `GLM_FLASHINFER_PRETUNED_ONLY` | 未设置 | 未设置，使用原生自动调优。 |
| Store紧凑布局 `store_compact_group_payloads` | `false` | `true`。 |
| Store异步加载 `load_async` | `true` | `true`。 |
| Store异步查询 `lookup_async` | `false` | `true`。 |
| Tokenizer缓存容量 `VLLM_GLM53_TOKENIZER_CACHE_GIB` | `0`，关闭 | P为`2`、D为`16` GiB/API；共享模式实际L1另受64MiB限制。 |
| Tokenizer共享路径 `VLLM_GLM53_TOKENIZER_SHARED_PATH` | 空，使用私有缓存 | `/run/glm-tokenizer/segments.db`。 |
| Tokenizer共享容量 `VLLM_GLM53_TOKENIZER_SHARED_GIB` | 配置共享路径时为`8` GiB | P为`15.5`、D为`15.9375` GiB/台。 |
| Tokenizer私有L1 `VLLM_GLM53_TOKENIZER_LOCAL_L1_MIB` | 共享模式为`64` MiB/API | `64`；共享池加全部L1共16GiB/台。 |
| Tokenizer代际文件 `VLLM_GLM53_TOKENIZER_CACHE_EPOCH_FILE` | 空 | `/run/glm-tokenizer/epoch`。 |
| 目的HCA亲和 `MC_ENABLE_DEST_DEVICE_AFFINITY` | 未设置 | 仅P设为`1`。 |
| Master目标大块大小 `MC_EVICTION_FIT_REQUEST_BYTES` | `0`，关闭 | `60318720`字节。 |
| Master空闲槽阈值 `MC_EVICTION_FIT_MIN_SLOTS` | `64` | `64`。 |
| Master检查间隔 `MC_EVICTION_FIT_CHECK_MS` | `1000` | `1000`毫秒。 |
| Router PD路由 `VLLM_ROUTER_PD_TOKEN_BALANCE_CONFIG` | 未设置 | `/run/token-balance.json`。 |
| Router耦合路由 `VLLM_ROUTER_COUPLED_KV_BALANCE` | 未设置 | 关闭。 |
| Router复用P分词结果 `VLLM_ROUTER_PD_REUSE_PROMPT_IDS` | 未设置，开启 | 默认开启；显式`0`关闭，`1`开启，需要新版Router。 |
| P末段准入预算 `VLLM_EXPERIMENTAL_HOT_TAIL_ADMISSION` | `0`，关闭 | 关闭。 |
| P末段预算日志 `VLLM_HOT_TAIL_ADMISSION_DIAGNOSTICS` | `0`，关闭 | 关闭。 |

共享分词容量按整机API数量计算：共享池加全部私有L1不得超过总预算。示例使用16GiB/台，调整API数量时相应调整共享池容量。

Master的目标大块大小按实际KV保存对象填写；60318720字节是本示例值，不是其他模型的固定要求。槽数阈值和检查间隔按业务压力选择，目标大小为0时关闭额外检查。

### FlashInfer调优

多卡调优键移除只随rank变化的字段，保留并行大小、专家数量和形状；匹配的调优结果按配置哈希及DP/world rank持久化。普通启动保留自动调优。`GLM_FLASHINFER_PRETUNED_ONLY=1`只加载兼容结果，缺失或不匹配时报错；GPU、库版本或并行配置变化时重新调优。

### 可选Mamba末段容量预算

`ec48a62d`为异步加载请求登记未来CoW状态块需求，避免其他请求先耗尽余量；仅因预算不足时延后当前请求，让已有owner继续。它不修改块引用、Store或GPU栅栏，也不关闭async、FP8、MTP或CUDA graph。

适用条件：原生异步调度、前缀缓存、align模式Mamba、单DCP/PCP；MultiConnector子项为Mooncake PD producer和异步紧凑Store。只处理首次、纯外部异步加载且在同一物理末块的请求，不覆盖local hit、恢复、跨块增长或checkpoint等其他路径。

使用包含该调度源码的新P镜像时，设置`VLLM_EXPERIMENTAL_HOT_TAIL_ADMISSION=1`；D不设置。可另设`VLLM_HOT_TAIL_ADMISSION_DIAGNOSTICS=1`记录每个scheduler最多三类首事件。变量在scheduler创建时读取；关闭需按排空、新engine ID流程重启P。默认关闭时不加载helper或增加predictor调用。

日志`stop_required=true`表示预算元数据契约失败，应停止并检查该轮。该功能不保证全局无抢占；已有验证中抢占减少，但整体吞吐未提高，需按目标业务验证收益后选择启用。

## 10. 从源码构建

| 提交 | 构建入口 | 作用 |
| --- | --- | --- |
| `963215c6` | `integrations/glm53flash/manifest.json`、`prepare_sources.py`、`Dockerfile`、`Dockerfile.router` | 固定原生vLLM、FlashInfer及Router源码，应用硬件、缓存、调优和路由补丁。 |
| `2a496cff` | `integrations/glm53flash/Dockerfile.shared-tokenizer` | 在匹配基底上加入三个分词缓存模块，不包含后续PUT或调度代码。 |
| `e8b3717c` | Mooncake Store的`store/worker.py` | 将PUT压力恢复纳入对应worker；Store SDK不变。 |
| `58d3b2cb` | P容器环境变量 | 使用已有SDK的目的网卡亲和，无需重编译推理算子。 |
| `95f4f832`、`bdc687f4` | `integrations/glm53flash/build-mooncake-master.sh`、`Dockerfile.mooncake-master` | 仅构建Master，包含实际占用计账和可选的大块空间检查；命令见本文第2节。 |
| `ec48a62d` | `vllm/v1/core/sched/hot_tail_budget.py`、`scheduler.py`、`kv_cache_manager.py`、`vllm/envs.py` | 在新推理镜像中纳入可选预算；仅设置环境变量不会让旧镜像获得该功能。 |

还原固定集成源码时，使用尚不存在的输出目录：

~~~bash
.venv/bin/python integrations/glm53flash/prepare_sources.py sources
git clone --depth 1 --branch v0.29.0 https://github.com/vllm-project/vllm.git sources/vllm-rust-v029
(cd sources/router && cargo build --locked --release --bin vllm-router)
~~~

`prepare_sources.py`还原manifest固定的集成快照，后续独立功能按上表纳入对应镜像。原生扩展目标为`_C_stable_libtorch`；Router使用原生Rust二进制。构建时记录源码、补丁、ELF及镜像哈希，已测镜像ID不会随Git历史整理而改变。

构建包含默认分词结果复用的新Router时，使用Router的GLM分支，而不是上述归档快照：

~~~bash
git clone --single-branch --branch vllm-0.32.0-0.1.16-glm5.3-flash \
  https://github.com/tokmo-infra/router.git router-native
git clone --depth 1 --branch v0.29.0 https://github.com/vllm-project/vllm.git vllm-rust-v029
(cd router-native && cargo build --locked --release --bin vllm-router)
mkdir router-runtime
cp router-native/target/release/vllm-router router-runtime/vllm-router
cat > router-runtime/Dockerfile <<'EOF'
FROM sha256:21c9cfa19e348f84a99102ab02922d5cd3a1ef6d659881cac09e9d6479c4d07b
COPY --chmod=755 vllm-router /usr/local/bin/vllm-router
EOF
docker build -t local/glm53flash-router:native-ids-local router-runtime
# 新构建的镜像ID会不同；通过PD和L2验收后显式使用自己的标签。
# export ROUTER_IMAGE=local/glm53flash-router:native-ids-local
~~~
