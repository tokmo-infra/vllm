# GLM5.3 Flash 1103 · 4P4D 部署

本文集中给出已实际跑通的 4P4D 启动配置。4P4D 按主机计数：4 台 Prefill 主机、4 台 Decode 主机；实际为 **8P + 4D、64 rank**。所有命令在目标主机的 root Bash 会话执行，适用于已分配给本次部署的 GPU、端口和目录。

## 1. 拓扑与固定参数

| 主机（服务网 IP 示例） | 实例 | GPU / DEP / EP | API / bootstrap / DP RPC 端口 |
| --- | --- | --- | --- |
| 81–84（11.8.0.81–84） | 每台 p0、p1，共 8P | p0：GPU0–3，p1：GPU4–7；每实例 TP1 / DP4 / EP4 | p0：18483 / 19083 / 19550；p1：18493 / 19093 / 19650 |
| 85–88（11.8.0.85–88） | 每台 d0，共 4D | GPU0–7；每实例 TP1 / DP8 / EP8 | 18483 / 19083 / 19550 |
| 81–88 | 每台一个 Store | 每台共享 750 GiB，合计 6000 GiB | RPC 61244；HTTP 61245 |
| 86 | patched Master、Router | Router 16 CPU / 16 GiB | Master RPC / metadata / metrics：61241 / 61242 / 61243；Router API / metrics：18300 / 18301 |

| 项目 | P | D |
| --- | --- | --- |
| 每 DP 的 seq / batch tokens | 24 / 2048 | 32 / 1024 |
| CUDA graph | 原生 native auto，实际解析上限 192 | FULL_DECODE_ONLY，上限 128 |
| API 进程 | 4/实例，即 8/台 | 1/实例，即 1/台 |
| Tokenizer 共享池 | 15.5 GiB/台，p0/p1 共用 | 15.9375 GiB/台 |
| Tokenizer 私有 L1 | 64 MiB/API，共 0.5 GiB/台 | 64 MiB/API，共 0.0625 GiB/台 |
| Tokenizer 总预算 | 16 GiB/台 = 2 GiB/GPU，已含 L1 | 16 GiB/台 = 2 GiB/GPU，已含 L1 |
| 目的 HCA 亲和 | `MC_ENABLE_DEST_DEVICE_AFFINITY=1` | `MC_ENABLE_DEST_DEVICE_AFFINITY=1` |
| 末段准入预算（P 抢占优化） | 部署显式设 1；引擎代码默认仍为 0 | 沿用代码默认 0 |
| KV 接收线程 | `VLLM_MOONCAKE_LOAD_RECV_THREADS=4` | 沿用原生配置 |

两端保持原生 async、FP8 KV、MTP3、fullISL、scheduler 水位 0、TP1、GPU utilization 0.95、prefix match unit 256、Mamba align、原生 allgather_reducescatter。CLI block size 为 64，GLM 混合缓存实际解析物理页为 8960；max model len 使用 auto。P 保留原生图选择，启动后的上限应为 192。

Tokenizer 的 `CACHE_GIB` 参数 P=2、D=16 是兼容参数；共享模式的整机预算以共享池加全部 API 的 L1 计算，不能再额外加一份 16 GiB。各模型主机应有足够 tmpfs 和内存；Store 新分配前参考可用内存大于 850 GiB，全部服务启动后宿主仍至少保留 16 GiB 可用内存。

## 2. 源码与镜像

引擎运行源码为 `5aeb9bd97bc09928d6f406fbd20b17b86f2023dd`。以下是跑通产物的**本地 image ID 指纹**，不是 registry manifest digest，也不是公开可 pull 的地址：

| 角色 | 已核验指纹 / 来源 |
| --- | --- |
| P | `sha256:c27716e4b3c48c745f6a172851b2b8818d03cdfa488e0ae690599c9aa4fde5b2` |
| D85 / D86 / D88 | `sha256:449fddc45480878a998875eb69d1182c49c833e465c68bf6f0f290df5da2b28f` |
| D87 | `sha256:e0bf46ee70ecb588e2c3daab8b451b91cfa42046a700b7db4cdd97568bb14a29`；通过 `D87_IMAGE` 指定。 |
| Store | `sha256:bd6c77d709b8225ebf7b4a9a3890509a4e2df87af846fcc7230b65a7a57d16f7` |
| patched Master | `sha256:8c7b3faf37167405712c29da0cef1cc330f62a704db5ebd11f175d6646a6a27a` |
| Router | `sha256:c07f71ef603b3c28c2f2b48e0657c4c96fcee285428dcd64747ed53a3d3aeec5`；二进制 SHA256：`6c51e70abc6a668a716b4738f448b5c2034559d63c3fe2c18d73352a95f99448`。 |

在各目标主机预先导入或构建所需镜像，设置对应的 `P_IMAGE`、`D_IMAGE`、`D87_IMAGE`、`STORE_IMAGE`、`MASTER_IMAGE`、`ROUTER_IMAGE`。变量可使用已导入的完整 image ID 或本机可用的镜像名。下面的 `--pull=never` 要求镜像先就绪；跨主机分发时应在各节点核对 image ID。

引擎增量构建封装为 `integrations/glm53flash/Dockerfile`，通过 `BASE_IMAGE` 传入匹配的原生基底；其校验清单见 `integrations/glm53flash/release.json`。Master 使用 `integrations/glm53flash/build-mooncake-master.sh` 和 `Dockerfile.mooncake-master`，基于 Mooncake `719735896c86b56fabec6cf3e825fb2ea640597a`，包含实际占用计账与大块空间回收两个补丁。Store SDK 保持配套版本。

Router prompt ID 复用实现移植自 `08d7195`，默认开启，本文仍显式传 `VLLM_ROUTER_PD_REUSE_PROMPT_IDS=1`。它与 P 末段准入预算是两个独立开关。

### 可选：在构建机生成镜像

已有已验收产物时，直接设置上述运行镜像变量即可。自行构建时在仓库根目录执行：`P_IMAGE`、`D_IMAGE`、`D87_IMAGE`、`MASTER_IMAGE` 此时须填写可供 `docker build -t` 使用的镜像名/tag；各 `*_BASE_IMAGE` 填写构建机已有的匹配原生基底引用。

引擎 Dockerfile 的构建 ARG 名为 `BASE_IMAGE`；它读取 `release.json` 的 `base_validation_files` 校验基底，覆盖 `image_overlay_files` 所列文件，并保留 `runtime_source_files` 供运行源码核验。`historical_images` 记录的是原生历史基底指纹，不是本次 4P4D 的运行镜像指纹。构建结果需重新记录 ID，不能假定与现场产物 ID 相同。

~~~bash
# 在包含 integrations/glm53flash 的仓库根目录执行。
: "${P_IMAGE:?填写 P 构建目标 tag}" "${P_BASE_IMAGE:?填写匹配 P 原生基底}"
: "${D_IMAGE:?填写 D 构建目标 tag}" "${D_BASE_IMAGE:?填写匹配 D 原生基底}"
: "${D87_IMAGE:?填写 D87 构建目标 tag}" "${D87_BASE_IMAGE:?填写匹配 D87 原生基底}"
docker build -f integrations/glm53flash/Dockerfile \
  --build-arg BASE_IMAGE="$P_BASE_IMAGE" -t "$P_IMAGE" .
docker build -f integrations/glm53flash/Dockerfile \
  --build-arg BASE_IMAGE="$D_BASE_IMAGE" -t "$D_IMAGE" .
docker build -f integrations/glm53flash/Dockerfile \
  --build-arg BASE_IMAGE="$D87_BASE_IMAGE" -t "$D87_IMAGE" .

# 准备好固定版本的本地 Mooncake 源码及其 CMake/Ninja 构建依赖。
: "${MOONCAKE_SOURCE:?填写本地 Mooncake 源码绝对路径}"
: "${MASTER_IMAGE:?填写 patched Master 构建目标 tag}"
: "${MASTER_BASE_IMAGE:?填写匹配 Store SDK 的 Master 基底引用}"
test "$(git -C "$MOONCAKE_SOURCE" rev-parse HEAD)" = \
  "$(jq -er '.commit' integrations/glm53flash/mooncake-master.json)"
bash integrations/glm53flash/build-mooncake-master.sh \
  "$MOONCAKE_SOURCE" "${MOONCAKE_BUILD_DIR:-/tmp/mooncake-1103-build}" \
  "$PWD/mooncake-master-runtime"
docker build -f integrations/glm53flash/Dockerfile.mooncake-master \
  --build-arg BASE_IMAGE="$MASTER_BASE_IMAGE" -t "$MASTER_IMAGE" .
~~~

Master 构建脚本的前三个位置参数依次为源码目录、构建目录、runtime 输出目录；输出必须位于此构建上下文的 `mooncake-master-runtime/`，与 Dockerfile 的 COPY 路径一致。引擎镜像中的 P 末段准入默认仍是 0，由后面的部署命令单独开启。Router 镜像使用包含上述来源功能的独立产物。

## 3. 每台主机的公共准备

需要 Docker、NVIDIA Container Toolkit、RDMA 驱动，以及 Bash、curl、jq。以下会话变量需在各目标主机设置；`NODE_ID` 分别填 81–88，服务 IP、网卡和路径按实际环境覆盖。P/D/Router 使用同一份 API key，通过交互输入或受控环境注入，不写入文档或版本库。

所有节点的 `MODEL_DIR` 应指向同一版本模型文件，包括 tokenizer 配置、tokenizer.json 和 chat template。Router 也挂载该目录；如业务覆盖 template，必须对齐 Router 与两端的渲染和分词语义。

~~~bash
test "$(id -u)" -eq 0
set -euo pipefail
umask 077
export NODE_ID=81  # 在各主机分别改为 81、82、83、84、85、86、87、88
case "$NODE_ID" in 81|82|83|84|85|86|87|88) ;; *) exit 1 ;; esac
export NODE_IP="${NODE_IP:-11.8.0.$NODE_ID}"
export MASTER_IP="${MASTER_IP:-11.8.0.86}"
export ROUTER_IP="${ROUTER_IP:-11.8.0.86}"
export P_HOSTS="${P_HOSTS:-11.8.0.81 11.8.0.82 11.8.0.83 11.8.0.84}"
export D_HOSTS="${D_HOSTS:-11.8.0.85 11.8.0.86 11.8.0.87 11.8.0.88}"
export MODEL_DIR="${MODEL_DIR:-/data/models/GLM-5.3-Flash-NVFP4}"
export MODEL_NAME="${MODEL_NAME:-glm-5.3-flash-nvfp4}"
export SERVICE_PREFIX="${SERVICE_PREFIX:-glm53flash-1103-4p4d}"
export BASE_WORK="${BASE_WORK:-/data/services/$SERVICE_PREFIX}"
export CACHE_NAMESPACE="${CACHE_NAMESPACE:-$SERVICE_PREFIX-prefix256}"
export TOKENIZER_DIR="${TOKENIZER_DIR:-/dev/shm/$SERVICE_PREFIX/tokenizer}"
export GID_INDEX="${GID_INDEX:-3}"
if [ "$NODE_ID" -le 84 ]; then
  export NET_IF="${NET_IF:-enp115s0f1np1}"
else
  export NET_IF="${NET_IF:-ens19np0}"
fi
export HCAS_ALL="${HCAS_ALL:-mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7}"
export HCAS_P0="${HCAS_P0:-mlx5_0,mlx5_1,mlx5_2,mlx5_3}"
export HCAS_P1="${HCAS_P1:-mlx5_4,mlx5_5,mlx5_6,mlx5_7}"
if [ -z "${API_KEY:-}" ]; then
  read -r -s -p 'API key: ' API_KEY; printf '\n'
fi
: "${API_KEY:?API key 必须非空}"
export API_KEY
mkdir -p "$BASE_WORK"
test -f "$MODEL_DIR/tokenizer.json"

# 同机全部 API 共用该目录；root-owned 可避免 SSH 用户退出时 RemoveIPC 清理。
install -d -m 0700 "$TOKENIZER_DIR"
if [ ! -e "$TOKENIZER_DIR/epoch" ]; then
  test -z "$(find "$TOKENIZER_DIR" -mindepth 1 -maxdepth 1 -print -quit)" || {
    echo '已有共享池但缺少 epoch，请先核对代际状态' >&2; exit 1;
  }
  (set -o noclobber; cat /proc/sys/kernel/random/uuid > "$TOKENIZER_DIR/epoch")
fi
test ! -L "$TOKENIZER_DIR/epoch"
test -s "$TOKENIZER_DIR/epoch"
chown 0:0 "$TOKENIZER_DIR" "$TOKENIZER_DIR/epoch"
chmod 0700 "$TOKENIZER_DIR"
chmod 0600 "$TOKENIZER_DIR/epoch"

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

同机 p0/p1 必须使用相同 Tokenizer 路径、epoch 和缓存命名空间。epoch 已存在时原样保留；重建共享池应先停用该池的所有 API，再统一更换代际。共享目录和 epoch 应由服务用户持有，本例服务用户为 root。部署前确认选用目录、容器名、GPU 和端口的归属。

## 4. patched Master：在 Master 主机执行

PUT 压力恢复在引擎发送路径生效，GET 租约刷新在读取路径生效；Master 的实际占用计账和大块空间回收必须使用配套 patched Master。目标块 60318720 字节，目标可分配槽数 64，每 1000 ms 检查。`MC_EVICTION_FIT_REQUEST_BYTES` 的代码默认值为 0，本部署显式开启。Master 淘汰高水位 0.98 与模型 scheduler 水位 0 是不同配置。

~~~bash
if [ "$NODE_IP" = "$MASTER_IP" ]; then
  : "${MASTER_IMAGE:?设置已导入的 patched Master 镜像}"
  docker run -d --pull=never --name "$SERVICE_PREFIX-master" \
    --network host --runtime=runc --memory 4g \
    -e NVIDIA_VISIBLE_DEVICES=void \
    -e MC_TCP_BIND_ADDRESS="$MASTER_IP" \
    -e MOONCAKE_REQUESTER_LOCAL_HOSTNAME="$MASTER_IP" \
    -e MC_EVICTION_FIT_REQUEST_BYTES=60318720 \
    -e MC_EVICTION_FIT_MIN_SLOTS=64 -e MC_EVICTION_FIT_CHECK_MS=1000 \
    --entrypoint mooncake_master "$MASTER_IMAGE" \
    --rpc_port=61241 --enable_http_metadata_server=true \
    --http_metadata_server_host=0.0.0.0 --http_metadata_server_port=61242 \
    --metrics_port=61243 --eviction_high_watermark_ratio=0.98 \
    --allocation_strategy=local_first --logtostderr=true
fi
~~~

## 5. Store：81–88 每台执行一次

750 GiB 是每台机器一个共享 Store，不是每 GPU 一份。客户端的 global segment 为 0，只注册 2 GiB 本地传输缓冲；Store 的 local buffer 为 0。

~~~bash
: "${STORE_IMAGE:?设置已导入的 Store 镜像}"
LIBCUDA_PATH=$(readlink -f /lib/x86_64-linux-gnu/libcuda.so.1)
test -f "$LIBCUDA_PATH"
docker run -d --pull=never --name "$SERVICE_PREFIX-store$NODE_ID" \
  --init --network host --ipc host --runtime=runc \
  --memory 832g --memory-swap 832g --ulimit memlock=-1:-1 \
  --device /dev/infiniband --cap-add IPC_LOCK --cap-add SYS_NICE \
  -e NVIDIA_VISIBLE_DEVICES=void -e MC_GID_INDEX="$GID_INDEX" \
  -e MC_TCP_BIND_ADDRESS="$NODE_IP" -e MOONCAKE_REQUESTER_LOCAL_HOSTNAME="$NODE_IP" \
  -e NCCL_SOCKET_IFNAME="$NET_IF" -e GLOO_SOCKET_IFNAME="$NET_IF" \
  -v "$LIBCUDA_PATH:/opt/host-driver/libcuda.so.1:ro" \
  -e LD_LIBRARY_PATH=/opt/host-driver:/usr/local/cuda/lib64 \
  --entrypoint mooncake_client "$STORE_IMAGE" \
  --master_server_address="$MASTER_IP:61241" \
  --metadata_server="http://$MASTER_IP:61242/metadata" \
  --protocol=rdma --host="$NODE_IP" --port=61244 \
  --device_names="$HCAS_ALL" --global_segment_size=805306368000 \
  --local_buffer_size=0 --enable_http_server=true --http_port=61245 \
  --enable_offload=false --logtostderr=true
curl -fsS "http://$NODE_IP:61245/health"
~~~

## 6. 模型实例：81–84 启动 p0/p1，85–88 启动 d0

先在各主机定义同一个启动函数，再按节点角色调用。所有节点使用相同的 `CACHE_NAMESPACE`；每实例每次启动生成独立 engine ID。P 用 Mooncake producer 加 Store kv_both，D 用 Mooncake consumer 加 Store consumer；Store 异步 lookup/load 与 compact group payload 保持开启。

~~~bash
start_model() {
  local slot="$1" image gpu_request hcas port bootstrap rpc dp apis
  local seq batch shared_gib cache_gib pd_role store_role role_work engine_id kv ib_hca
  local -a role_env=() graph_args=()
  case "$slot" in
    p0|p1)
      image="${P_IMAGE:?设置已导入的 P 镜像}"
      dp=4; apis=4; seq=24; batch=2048; shared_gib=15.5; cache_gib=2
      pd_role=kv_producer; store_role=kv_both
      role_env=(-e VLLM_MOONCAKE_LOAD_RECV_THREADS=4
                -e VLLM_EXPERIMENTAL_HOT_TAIL_ADMISSION=1
                -e VLLM_HOT_TAIL_ADMISSION_DIAGNOSTICS=0)
      if [ "$slot" = p0 ]; then
        gpu_request='"device=0,1,2,3"'; hcas="$HCAS_P0"
        port=18483; bootstrap=19083; rpc=19550
      else
        gpu_request='"device=4,5,6,7"'; hcas="$HCAS_P1"
        port=18493; bootstrap=19093; rpc=19650
      fi
      ;;
    d0)
      if [ "$NODE_ID" = 87 ]; then
        image="${D87_IMAGE:?设置已导入的 D87 镜像}"
      else
        image="${D_IMAGE:?设置已导入的 D 镜像}"
      fi
      dp=8; apis=1; seq=32; batch=1024; shared_gib=15.9375; cache_gib=16
      pd_role=kv_consumer; store_role=kv_consumer
      gpu_request='"device=0,1,2,3,4,5,6,7"'; hcas="$HCAS_ALL"
      port=18483; bootstrap=19083; rpc=19550
      graph_args=(--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":128}')
      ;;
    *) return 1 ;;
  esac
  ib_hca="=$(printf '%s' "$hcas" | sed 's/,/:1,/g'):1"
  role_work="$BASE_WORK/$slot"
  mkdir -p "$role_work/cache"
  write_mc_config "$role_work/mooncake.json" "$hcas"
  engine_id="$SERVICE_PREFIX-$slot-$NODE_ID-$(cat /proc/sys/kernel/random/uuid)"
  kv=$(cat <<EOF
{
  "kv_connector":"MultiConnector","kv_role":"kv_both","engine_id":"$engine_id",
  "kv_load_failure_policy":"fail",
  "kv_connector_extra_config":{"connectors":[
    {"kv_connector":"MooncakeConnector","kv_role":"$pd_role",
     "kv_connector_extra_config":{"mooncake_protocol":"rdma","device_name":"$hcas"}},
    {"kv_connector":"MooncakeStoreConnector","kv_role":"$store_role",
     "kv_connector_extra_config":{"cache_prefix":"$CACHE_NAMESPACE",
       "load_async":true,"lookup_async":true,"store_compact_group_payloads":true}}
  ]}
}
EOF
)
  docker run -d --pull=never --name "$SERVICE_PREFIX-$slot-$NODE_ID" \
    --init --network host --shm-size 64g --memory 96g --memory-swap 96g \
    --gpus "$gpu_request" --device /dev/infiniband \
    --cap-add IPC_LOCK --cap-add SYS_NICE --ulimit memlock=-1:-1 \
    --ulimit stack=67108864:67108864 \
    -v "$MODEL_DIR:/model:ro" -v "$role_work:/work" \
    -v "$role_work/cache:/root/.cache" \
    -v "$TOKENIZER_DIR:/run/glm-tokenizer:rw" \
    -e VLLM_GLM53_TOKENIZER_SHARED_PATH=/run/glm-tokenizer/segments.db \
    -e VLLM_GLM53_TOKENIZER_SHARED_GIB="$shared_gib" \
    -e VLLM_GLM53_TOKENIZER_CACHE_GIB="$cache_gib" \
    -e VLLM_GLM53_TOKENIZER_LOCAL_L1_MIB=64 \
    -e VLLM_GLM53_TOKENIZER_CACHE_EPOCH_FILE=/run/glm-tokenizer/epoch \
    -e MC_TCP_BIND_ADDRESS="$NODE_IP" -e MOONCAKE_REQUESTER_LOCAL_HOSTNAME="$NODE_IP" \
    -e MC_ENABLE_DEST_DEVICE_AFFINITY=1 \
    -e VLLM_HOST_IP="$NODE_IP" -e NCCL_SOCKET_IFNAME="$NET_IF" \
    -e GLOO_SOCKET_IFNAME="$NET_IF" -e NCCL_IB_HCA="$ib_hca" \
    -e NCCL_IB_GID_INDEX="$GID_INDEX" -e MC_GID_INDEX="$GID_INDEX" -e NCCL_DEBUG=INFO \
    -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_SERVER_DEV_MODE=1 \
    -e MOONCAKE_CONFIG_PATH=/work/mooncake.json \
    -e VLLM_MOONCAKE_BOOTSTRAP_PORT="$bootstrap" -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120 \
    ${role_env[@]+"${role_env[@]}"} \
    --entrypoint vllm "$image" serve /model \
    --served-model-name "$MODEL_NAME" --host "$NODE_IP" --port "$port" \
    --api-key "$API_KEY" --api-server-count "$apis" --language-model-only \
    --tensor-parallel-size 1 --data-parallel-size "$dp" --data-parallel-size-local "$dp" \
    --data-parallel-backend mp --data-parallel-rpc-port "$rpc" \
    --enable-expert-parallel --enable-ep-weight-filter \
    --max-model-len auto --max-num-seqs "$seq" --max-num-batched-tokens "$batch" \
    --gpu-memory-utilization 0.95 --kv-cache-dtype fp8 \
    --block-size 64 --prefix-match-unit 256 --mamba-cache-mode align \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --kv-transfer-config "$kv" --scheduler-reserve-full-isl \
    --enable-prefix-caching --reasoning-parser glm47 \
    --enable-auto-tool-choice --tool-call-parser glm47 \
    --safetensors-load-strategy lazy --enable-cumem-allocator --seed 42 \
    --enable-prompt-tokens-details --enable-force-include-usage --numa-bind \
    ${graph_args[@]+"${graph_args[@]}"}
}

if [ "$NODE_ID" -le 84 ]; then
  start_model p0
  start_model p1
else
  start_model d0
fi
~~~

末段准入预算用于异步加载后需要额外 CoW 状态块的特定 producer 请求，保护余量以减少抢占重算。引擎 `VLLM_EXPERIMENTAL_HOT_TAIL_ADMISSION` 默认 0，本部署只在 P 的命令里设为 1；诊断开关 `VLLM_HOT_TAIL_ADMISSION_DIAGNOSTICS` 保持 0。FP8、MTP3、原生异步、图选择和 fullISL 与该开关分别配置。

## 7. Router：在 Router 主机执行

等待全部 12 个模型实例完成启动，读取各实例**实际** GPU KV 容量生成路由配置；P max_seqs=24，D max_seqs=32，kv_fraction=0.85。Router 使用同模型 `/model/tokenizer.json` 及一致的 chat template，直接连接原生 vLLM HTTP API。复用 prompt IDs 的部署开关显式设为 1，资源固定 16 CPU / 16 GiB。

~~~bash
if [ "$NODE_IP" = "$ROUTER_IP" ]; then
  : "${ROUTER_IMAGE:?设置含 prompt ID 复用功能的 Router 镜像}"
  read -r -a p_hosts <<< "$P_HOSTS"
  read -r -a d_hosts <<< "$D_HOSTS"
  test "${#p_hosts[@]}" -eq 4
  test "${#d_hosts[@]}" -eq 4
  workers='{}'
  routes=()
  add_worker() {
    local url="$1" max_seqs="$2" kv_tokens
    kv_tokens=$(curl -fsS -H "Authorization: Bearer $API_KEY" \
      "$url/server_info?config_format=json" |
      jq -er '.vllm_config.cache_config.kv_cache_size_tokens | numbers | select(. > 0)')
    workers=$(jq -c --arg url "$url" --argjson kv "$kv_tokens" \
      --argjson seq "$max_seqs" \
      '. + {($url): {kv_tokens: $kv, max_seqs: $seq}}' <<< "$workers")
  }
  for ip in "${p_hosts[@]}"; do
    add_worker "http://$ip:18483" 24
    add_worker "http://$ip:18493" 24
    routes+=(--prefill "http://$ip:18483" 19083 --prefill "http://$ip:18493" 19093)
  done
  for ip in "${d_hosts[@]}"; do
    add_worker "http://$ip:18483" 32
    routes+=(--decode "http://$ip:18483")
  done
  test "$(jq 'length' <<< "$workers")" -eq 12
  jq -n --argjson workers "$workers" \
    '{tokenizer_path:"/model/tokenizer.json",kv_fraction:0.85,workers:$workers}' \
    > "$BASE_WORK/token-balance.json"

  docker run -d --pull=never --name "$SERVICE_PREFIX-router" \
    --runtime=runc --network host --cpus 16 --memory 16g --memory-swap 16g \
    -v "$MODEL_DIR:/model:ro" \
    -v "$BASE_WORK/token-balance.json:/run/token-balance.json:ro" \
    -e VLLM_ROUTER_PD_TOKEN_BALANCE_CONFIG=/run/token-balance.json \
    -e VLLM_ROUTER_PD_REUSE_PROMPT_IDS=1 \
    -e NVIDIA_VISIBLE_DEVICES=void -e OPENAI_API_KEY="$API_KEY" \
    -e MC_TCP_BIND_ADDRESS="$ROUTER_IP" -e MOONCAKE_REQUESTER_LOCAL_HOSTNAME="$ROUTER_IP" \
    --entrypoint /usr/local/bin/vllm-router "$ROUTER_IMAGE" \
    --vllm-pd-disaggregation --kv-connector mooncake \
    "${routes[@]}" \
    --prefill-policy cache_aware --decode-policy cache_aware \
    --balance-abs-threshold 4 --balance-rel-threshold 1.5 \
    --intra-node-data-parallel-size 1 --host "$ROUTER_IP" --port 18300 \
    --prometheus-host "$ROUTER_IP" --prometheus-port 18301 \
    --disable-retries --worker-startup-timeout-secs 600 \
    --worker-startup-check-interval 5 --log-level info --api-key "$API_KEY"
fi
~~~

## 8. 启动核验

启动顺序为 Master → 全部 Store → D / P 模型实例 → Router。保留模型启动日志，确认 P 原生图上限 192、D 图上限 128；原生 FlashInfer 预热和匹配的调优缓存复用保持生效。

在 Router 主机运行下列只读检查。模型预热期间可待其就绪后重试；64 个健康 worker 是 64 个 DP rank，而不是 64 个独立 HTTP origin。

~~~bash
for ip in $P_HOSTS $D_HOSTS; do
  curl -fsS "http://$ip:61245/health"
done
curl -fsS "http://$MASTER_IP:61243/metrics" |
  awk '$1 == "master_total_capacity_bytes" {print; ok=($2 == 6442450944000)} END {exit !ok}'

check_model() {
  local url="$1" dp="$2" seq="$3" batch="$4" graph="$5"
  curl -fsS -H "Authorization: Bearer $API_KEY" \
    "$url/server_info?config_format=json" |
    jq -e --argjson dp "$dp" --argjson seq "$seq" --argjson batch "$batch" --argjson graph "$graph" '
      .vllm_config |
      .scheduler_config.async_scheduling == true and
      .scheduler_config.scheduler_reserve_full_isl == true and
      .scheduler_config.watermark == 0 and
      .scheduler_config.max_num_seqs == $seq and
      .scheduler_config.max_num_batched_tokens == $batch and
      .parallel_config.tensor_parallel_size == 1 and
      .parallel_config.data_parallel_size == $dp and
      .parallel_config.all2all_backend == "allgather_reducescatter" and
      .cache_config.cache_dtype == "fp8" and
      .speculative_config.num_speculative_tokens == 3 and
      .compilation_config.max_cudagraph_capture_size == $graph'
}
for ip in $P_HOSTS; do
  check_model "http://$ip:18483" 4 24 2048 192
  check_model "http://$ip:18493" 4 24 2048 192
done
for ip in $D_HOSTS; do
  check_model "http://$ip:18483" 8 32 1024 128
done
curl -fsS -H "Authorization: Bearer $API_KEY" "http://$ROUTER_IP:18300/workers" |
  jq -e '.total == 64 and (.workers | length == 64) and all(.workers[]; .is_healthy == true)'
~~~

各模型主机再确认宿主 `MemAvailable` 至少 16 GiB，Tokenizer 目录/epoch 的 uid、gid 为 0，p0/p1 的挂载指向同一共享池。P 启动日志应有每个 DP scheduler 的 `P_HOT_TAIL_BUDGET init`，4 条/实例，`enabled=True`、`progress_enabled=False`。Router image、复用开关、CPU/内存限制和模型文件身份应与部署输入一致。
