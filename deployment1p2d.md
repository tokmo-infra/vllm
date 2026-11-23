# GLM-5.3 Flash NVFP4 · 1P2D 部署

**首次启动：D1 上的 Mooncake Master → P 的两个实例和 D1/D2 各一个实例预热 → 三台 Mooncake Store → P 上的 Router。** 模型加载、FlashInfer 调优和 CUDA 图预热完成后，再分配大容量 L2。

**编译缓存：** 首次启动成功后，后续在同一台机器、同一 GPU 槽位和兼容的镜像及配置下复用编译缓存盘上 `COMPILE_CACHE_ROOT` 下的实例目录。每个目录都挂到对应容器的 `/root/.cache`，保存 FlashInfer、B12X 和 vLLM 的编译与调优产物。清分词缓存或 Mooncake KV 时，保留这些目录。

| 节点 / 实例 | 默认主机路径 |
| --- | --- |
| P / p0 | `$COMPILE_CACHE_ROOT/p0` |
| P / p1 | `$COMPILE_CACHE_ROOT/p1` |
| D1、D2 / d0 | 各自本机的 `$COMPILE_CACHE_ROOT/d0` |

**重启实例：** 停止放流并排空在途请求，保留健康的 D、Master 和 Store。从原槽位目录重建需要调整的 P 实例。两个 P 的 `/health` 都通过、两个 bootstrap 的 `/query` 各注册 4 个 DP 后，重启 P 上的 Router 让它重新读取 P 的 `engine_id`；沿用原 IP 和端口时也是这个顺序。核对 Router 24/24 健康，并跑一条真实 P→D 请求再恢复流量。

**核对地址：** 以 `deploy.env` 为准。模型监听、`VLLM_HOST_IP`、Mooncake 连接、Router 路由和验收命令使用同一份地址与端口。用 `/query` 检查 P 报出的 worker 地址。

| 角色 | 进程 | GPU / 每 DP 参数 | 端口配置 |
| --- | --- | --- | --- |
| P | p0、p1、Router | p0 GPU 0–3，p1 GPU 4–7；各 DP4/EP4，seq 24、batch 2048、原生图上限 192，4 API 进程/实例 | `P0_*_PORT`、`P1_*_PORT`、`ROUTER_PORT`、`ROUTER_METRICS_PORT` |
| D1 | d0、Mooncake Master | GPU 0–7；DP8/EP8，seq 32、batch 1024、解码图上限 128，1 API 进程 | `D1_*_PORT`、`MASTER_*_PORT` |
| D2 | d0 | 同 D1 的 d0 | `D2_*_PORT` |
| P / D1 / D2 | 每台一个 Mooncake Store | L2 700 / 750 / 750 GiB；L3 每台 2.5 TB | `STORE_RPC_PORT`、`STORE_HEALTH_PORT` |

| 字段 | 含义 |
| --- | --- |
| `--kv-cache-dtype nvfp4` | P、D 使用同一种 KV 格式。主干稀疏 MLA 使用 304 B NVFP4 记录，索引、KDA 和 MTP 仍用各自原有格式。既有 FP8 部署切换时，两个角色一起换镜像和 KV 类型，并使用新的 Mooncake 缓存命名空间。 |
| `--block-size 64` / `--prefix-match-unit 256` | CLI block 64，前缀匹配仍为 256。混合缓存的调度页由引擎计算，这套配置解析为 14,848 token。 |
| `VLLM_GLM53_P_MTP_CONTEXT_ONLY=1` | P 只加载产生 MTP 交接状态所需的草稿模型权重；D 使用完整 MTP3。P 的 `VLLM_GLM53_P_READY_WAIT=1` 只记录 KV 读完到首次下发的等待。 |
| NVFP4 MoE | 两侧启用融合、FC2 staging 和权重向量化；P 另开扩展预取。引擎沿用 TP1、原生异步、full-ISL、CUDA 图和默认后端；GPU 利用率 0.95。每个 vLLM 容器内存上限 64 GiB，禁用 Swap。 |
| 图片输入 | P、D 均加载视觉模块，推理端不配置额外的每请求图片张数上限。图片尺寸、总上下文和显存仍受模型与引擎约束。 |
| 分词缓存 | 每台模型主机共享一个 16 GiB 池，每个 API 进程另有 128 MiB 私有 L1（每个 P 实例 4 个 API 进程，每个 D 实例 1 个）。两个 P 实例共用 P 主机的池；D1 和 D2 各有自己的池。D 复用 P 返回的 prompt IDs。运行时 `VLLM_GLM53_TOKENIZER_CACHE_GIB` 在 P 为 2、D 为 16，共享池上限均为 16。 |
| Router | P 为 `cache_aware`，D 为 `round_robin`；开启 prompt ID 复用和 P 单 token 完成交接。32 GiB 内存上限、禁用 Swap，CPU 核数沿用宿主机；客户端访问 `ROUTER_BASE_URL`，推理请求无需客户端 key；Router 到 P/D 使用内部 API key。 |
| Mooncake | P→D 使用 RDMA；Store 异步查询与读回，启用紧凑 KV 及旧检查点回收（仅 P）。Master 在 D1，淘汰高水位为 95%；L3 按需写盘，Store 使用 LRU。 |

## 构建镜像

Linux x86_64 构建机，在本分支根目录执行。两种构建产出同一套应用镜像。FlashInfer 主包为 0.7.1（含调优键补丁）；B12X 取 `1658f0bf` 的源码，只补入 B12X 和其私有锁模块，并使用本仓库的 NVFP4 读取补丁。Mooncake SDK、Master、Store 来自同一份 `71973589` 源码和配套补丁；Router 使用仓库内的 GLM 1103 配套源码快照。

### 基于已有的 vLLM 基镜像

基镜像为 `local/glm53flash-1103:source-67ca8f23`；构建时叠加 NVFP4 KV、MTP、Mooncake connector、调度观测及 API 兼容模块。此命令同时从源码构建 Router 和 Mooncake。已有配套 Router、Mooncake 镜像时，子命令 `overlay` 只构建引擎。

~~~bash
mkdir -p /data/build/glm53flash-1p2d
BUILD_PARENT=/data/build/glm53flash-1p2d BUILD_JOBS=8 \
  bash integrations/glm53flash/1p2d/build/build-all.sh base
~~~

### 从源码构建

两种构建方式均校验 vLLM 源码和覆盖层的文件哈希；Router 使用随本仓库提供的源码快照并独立校验。GLM-5.3-Flash 保持原厂始终思考行为，原生档位为 low、high、max。Responses 流式终态与实际生成状态一致。新版引擎镜像包含 Chat 工具终态和流式计量修复；此标签为构建目标，线上版本以实际容器为准。

Buildx 先构建 FlashInfer 和 Mooncake wheel，编译 vLLM CUDA/Rust 基底（SM120），再应用同一份 NVFP4 KV 覆盖层；Router 使用 Cargo 锁定依赖编译。基础 CUDA/PyTorch 镜像和系统依赖由构建机获取。`GLM_BUILDER` 可指定已有 Buildx builder，`GLM_BUILD_PROXY` 可传入依赖下载代理。

~~~bash
mkdir -p /data/build/glm53flash-1p2d
BUILD_PARENT=/data/build/glm53flash-1p2d BUILD_JOBS=8 \
  bash integrations/glm53flash/1p2d/build/build-all.sh source
~~~

| 用途 | 默认镜像 |
| --- | --- |
| P 和 D | `local/glm53flash-nvfp4kv:compat-api-v7-20261124` |
| Mooncake Master 和 Store | `local/glm53flash-mooncake:source-71973589` |
| Router | `router:glm53flash-api-v7-20261124` |

~~~bash
# 构建机
set -o pipefail
docker save local/glm53flash-nvfp4kv:compat-api-v7-20261124 \
  local/glm53flash-mooncake:source-71973589 router:glm53flash-api-v7-20261124 \
  | zstd -T4 -3 -o glm53flash-1p2d-images.tar.zst

# P / D1 / D2 导入镜像包
zstd -dc glm53flash-1p2d-images.tar.zst | sudo docker load
~~~

单张内联图片默认上限为 4 MiB（4,194,304 字节，按 Base64 解码后的图片大小计算）。超限 JPEG/PNG 在选路前压缩到上限内，P/D 使用同一份字节；无法压到上限或超限格式不受支持时返回 HTTP 400。上限可通过 `VLLM_ROUTER_GLM_INLINE_IMAGE_MAX_BYTES` 调整。纯文本、外链和上限内的图片原样转发。外链由 vLLM 按后端网络和媒体域名配置获取。Chat 的视频块使用 `video_url`，Responses 使用 `input_video`，Messages 及其 token 计数接口使用 `video`。内联视频由 Router 在选路前处理一次；P 和 D 收到相同内容。远端 URL 由引擎按媒体访问策略读取。当前模型每个 prompt 最多一段视频；验收的是画面输入，转码输出不含音轨。

## 启动服务

启动顺序：D1 Master → P 两个实例、D1/D2 各一个 D 实例 → 三台 Store → P Router。以下命令以 root Bash 在各节点的仓库根目录执行。同一台机器的后续命令使用同一个 Bash 会话。

### 1. 部署参数 · 三台主机

在三台机器保存同一份 `deploy.env`，把 `<...>` 换成实际值。端口是可替换的示例值，同机监听端口须各自独立；同一服务端口在不同机器可以相同。`P_IP`、`D1_IP`、`D2_IP` 填各机 RDMA IPv4。Router 与 P 同机，Master 与 D1 同机。

~~~bash
# deploy.env
P_IP='<P_RDMA_IP>'
D1_IP='<D1_RDMA_IP>'
D2_IP='<D2_RDMA_IP>'
P_NET_IF='<P_RDMA_IFACE>'
D1_NET_IF='<D1_RDMA_IFACE>'
D2_NET_IF='<D2_RDMA_IFACE>'

# 各实例 HTTP、KV bootstrap、DP RPC；按现场调整
P0_HTTP_PORT=18483; P0_BOOTSTRAP_PORT=19083; P0_DP_RPC_PORT=19550
P1_HTTP_PORT=18493; P1_BOOTSTRAP_PORT=19093; P1_DP_RPC_PORT=19650
D1_HTTP_PORT=18483; D1_BOOTSTRAP_PORT=19083; D1_DP_RPC_PORT=19550
D2_HTTP_PORT=18483; D2_BOOTSTRAP_PORT=19083; D2_DP_RPC_PORT=19550
ROUTER_PORT=18300; ROUTER_METRICS_PORT=18301
ROUTER_REQUEST_TIMEOUT_SECS=3600
MASTER_RPC_PORT=61241; MASTER_METADATA_PORT=61242; MASTER_METRICS_PORT=61243
STORE_RPC_PORT=61244; STORE_HEALTH_PORT=61245

# 下列监听地址与 RDMA 设备映射按现场填写
ROUTER_BIND_HOST=0.0.0.0
ROUTER_METRICS_BIND_HOST=0.0.0.0
MASTER_METADATA_BIND_HOST=0.0.0.0
P_GID_INDEX=3; D1_GID_INDEX=3; D2_GID_INDEX=3
P0_HCAS='mlx5_0,mlx5_1,mlx5_2,mlx5_3'
P1_HCAS='mlx5_4,mlx5_5,mlx5_6,mlx5_7'
D1_HCAS='mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7'
D2_HCAS='mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7'

# 仅记录密钥文件路径。三台文件内容一致，归 root 所有，权限 0600。
INTERNAL_API_KEY_FILE='<ABSOLUTE_PATH_TO_INTERNAL_API_KEY_FILE>'
SERVICE_PREFIX=glm53flash-1p2d
MODEL_DIR='/data/models/GLM-5.3-Flash-NVFP4'
SERVICE_ROOT='/data/services'
COMPILE_CACHE_ROOT='/var/cache/glm53flash-1p2d/d50c0828c8dd62'
L3_MOUNT='/data'
L2_P_GIB=700; L2_D1_GIB=750; L2_D2_GIB=750
~~~

在各机设置本机角色：P 执行 `NODE_ROLE=p`，D1 执行 `NODE_ROLE=d1`，D2 执行 `NODE_ROLE=d2`。随后载入公共参数。内部 key 用于 Router → vLLM，客户端调用 Router 的推理接口时无需提供 key。

~~~bash
set -euo pipefail
umask 077
source ./deploy.env
: "${NODE_ROLE:?请先设置 NODE_ROLE=p、d1 或 d2}"
case "$NODE_ROLE" in
  p)  NODE_IP="$P_IP"; NET_IF="$P_NET_IF"; GID_INDEX="$P_GID_INDEX"; NODE_HCAS="$P0_HCAS,$P1_HCAS" ;;
  d1) NODE_IP="$D1_IP"; NET_IF="$D1_NET_IF"; GID_INDEX="$D1_GID_INDEX"; NODE_HCAS="$D1_HCAS" ;;
  d2) NODE_IP="$D2_IP"; NET_IF="$D2_NET_IF"; GID_INDEX="$D2_GID_INDEX"; NODE_HCAS="$D2_HCAS" ;;
  *) echo "未知角色：$NODE_ROLE" >&2; exit 2 ;;
esac
ip -o -4 addr show dev "$NET_IF" |
  awk -v ip="$NODE_IP" '{split($4,a,"/"); if (a[1]==ip) ok=1} END {exit !ok}' || {
  echo "请核对 $NODE_ROLE 的 $NET_IF 是否配置了 $NODE_IP。" >&2; exit 1;
}

MASTER_IP="$D1_IP"
MASTER_RPC="$MASTER_IP:$MASTER_RPC_PORT"
MASTER_METADATA_URL="http://$MASTER_IP:$MASTER_METADATA_PORT/metadata"
MASTER_METRICS_URL="http://$MASTER_IP:$MASTER_METRICS_PORT/metrics"
P0_URL="http://$P_IP:$P0_HTTP_PORT"; P1_URL="http://$P_IP:$P1_HTTP_PORT"
D1_URL="http://$D1_IP:$D1_HTTP_PORT"; D2_URL="http://$D2_IP:$D2_HTTP_PORT"
ROUTER_URL="http://$P_IP:$ROUTER_PORT"
ROUTER_BASE_URL="$ROUTER_URL/v1"
BASE_WORK="$SERVICE_ROOT/$SERVICE_PREFIX"
CACHE_NAMESPACE="$SERVICE_PREFIX-nvfp4-prefix256"
TOKENIZER_DIR="/dev/shm/$SERVICE_PREFIX/tokenizer"
test -f "$INTERNAL_API_KEY_FILE"; test ! -L "$INTERNAL_API_KEY_FILE"
test "$(stat -c '%u:%g:%a' "$INTERNAL_API_KEY_FILE")" = '0:0:600'
INTERNAL_API_KEY=$(cat -- "$INTERNAL_API_KEY_FILE")
test -n "$INTERNAL_API_KEY"
MODEL_NAME=glm-5.3-flash
here="$PWD/integrations/glm53flash/1p2d"

P_IMAGE="${ENGINE_IMAGE:-local/glm53flash-nvfp4kv:compat-api-v7-20261124}"
D_IMAGE="$P_IMAGE"
MASTER_IMAGE="${MOONCAKE_IMAGE:-local/glm53flash-mooncake:source-71973589}"
STORE_IMAGE="$MASTER_IMAGE"
ROUTER_IMAGE="${ROUTER_IMAGE:-router:glm53flash-api-v7-20261124}"
~~~

### 2. Mooncake Master · D1

~~~bash
test "$NODE_ROLE" = d1
LIBCUDA_PATH=$(readlink -f /lib/x86_64-linux-gnu/libcuda.so.1)
test -f "$LIBCUDA_PATH"
docker run -d --pull=never --restart unless-stopped --name "$SERVICE_PREFIX-master" \
  --network host --runtime=runc --memory 4g \
  -e NVIDIA_VISIBLE_DEVICES=void \
  -e MC_TCP_BIND_ADDRESS="$MASTER_IP" \
  -v "$LIBCUDA_PATH:/opt/host-driver/libcuda.so.1:ro" \
  -e LD_LIBRARY_PATH=/opt/host-driver:/usr/local/cuda/lib64 \
  -e MOONCAKE_REQUESTER_LOCAL_HOSTNAME="$MASTER_IP" \
  -e MC_EVICTION_FIT_REQUEST_BYTES=60318720 \
  -e MC_EVICTION_FIT_MIN_SLOTS=64 -e MC_EVICTION_FIT_CHECK_MS=1000 \
  --entrypoint mooncake_master "$MASTER_IMAGE" \
  --rpc_port="$MASTER_RPC_PORT" --enable_http_metadata_server=true \
  --http_metadata_server_host="$MASTER_METADATA_BIND_HOST" --http_metadata_server_port="$MASTER_METADATA_PORT" \
  --metrics_port="$MASTER_METRICS_PORT" --eviction_high_watermark_ratio=0.95 \
  --enable_offload=true --offload_on_evict=true \
  --allocation_strategy=local_first --logtostderr=true

curl -fsS "$MASTER_METRICS_URL" >/dev/null
~~~

### 3. vLLM · P / D1 / D2

在三台模型主机上定义同一个启动函数。P 两个实例共用本机分词池；编译和调优缓存按实例保存在开头的系统盘路径。首次准备可将同镜像、同槽位的旧缓存复制到该目录；全新构建则创建空目录，让初次启动写入。

~~~bash
test -f "$MODEL_DIR/tokenizer.json"
# NUMA 包装脚本保存在本服务目录，由各实例只读挂载。
install -d -m 0700 "$BASE_WORK"
if ! cmp -s "$here/numa_wrapper.sh" "$BASE_WORK/numa_wrapper.sh"; then
  install -m 0755 "$here/numa_wrapper.sh" "$BASE_WORK/numa_wrapper.sh"
fi
install -d -m 0700 "$TOKENIZER_DIR"
if [[ ! -e "$TOKENIZER_DIR/epoch" ]]; then
  (set -o noclobber; cat /proc/sys/kernel/random/uuid > "$TOKENIZER_DIR/epoch")
fi
test ! -L "$TOKENIZER_DIR/epoch"; test -s "$TOKENIZER_DIR/epoch"
chown 0:0 "$TOKENIZER_DIR" "$TOKENIZER_DIR/epoch"
chmod 0700 "$TOKENIZER_DIR"
chmod 0600 "$TOKENIZER_DIR/epoch"

write_mc_config() {
  cat > "$1" <<EOF
{
  "mode": "standalone-store",
  "metadata_server": "$MASTER_METADATA_URL",
  "master_server_address": "$MASTER_RPC",
  "protocol": "rdma",
  "device_name": "$2",
  "global_segment_size": 0,
  "local_buffer_size": 2147483648,
  "enable_offload": true,
  "tenant_id": "default"
}
EOF
}

start_model() {
  local slot="$1" image gpu_request hcas port bootstrap rpc dp apis
  local seq batch shared_gib cache_gib pd_role store_role role_work engine_id kv ib_hca cache_dir
  local retire_checkpoints=false gpu_util=0.95
  local -a role_env=() graph_args=()
  case "$slot" in
    p0|p1)
      image="$P_IMAGE"
      dp=4; apis=4; seq=24; batch=2048; shared_gib=16; cache_gib=2
      gpu_util="${P_GPU_UTIL:-0.95}"; pd_role=kv_producer; store_role=kv_both; retire_checkpoints=true
      role_env=(-e VLLM_GLM53_P_MTP_CONTEXT_ONLY=1 -e VLLM_GLM53_P_READY_WAIT=1
                -e VLLM_GLM53_NVFP4_EXTENDED_PREFETCH=1
                -e VLLM_MOONCAKE_LOAD_RECV_THREADS=4
                -e VLLM_EXPERIMENTAL_HOT_TAIL_ADMISSION=1
                -e VLLM_HOT_TAIL_ADMISSION_DIAGNOSTICS=0)
      if [ "$slot" = p0 ]; then
        gpu_request='"device=0,1,2,3"'; hcas="$P0_HCAS"
        port="$P0_HTTP_PORT"; bootstrap="$P0_BOOTSTRAP_PORT"; rpc="$P0_DP_RPC_PORT"
      else
        gpu_request='"device=4,5,6,7"'; hcas="$P1_HCAS"
        port="$P1_HTTP_PORT"; bootstrap="$P1_BOOTSTRAP_PORT"; rpc="$P1_DP_RPC_PORT"
      fi
      ;;
    d0)
      image="$D_IMAGE"
      dp=8; apis=1; seq=32; batch=1024; shared_gib=16; cache_gib=16
      role_env=(-e VLLM_GLM53_P_MTP_CONTEXT_ONLY=0
                -e VLLM_GLM53_NVFP4_EXTENDED_PREFETCH=0)
      pd_role=kv_consumer; store_role=kv_consumer
      gpu_request='"device=0,1,2,3,4,5,6,7"'; hcas="$NODE_HCAS"
      case "$NODE_ROLE" in
        d1) port="$D1_HTTP_PORT"; bootstrap="$D1_BOOTSTRAP_PORT"; rpc="$D1_DP_RPC_PORT" ;;
        d2) port="$D2_HTTP_PORT"; bootstrap="$D2_BOOTSTRAP_PORT"; rpc="$D2_DP_RPC_PORT" ;;
        *) return 1 ;;
      esac
      graph_args=(--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":128}')
      ;;
    *) return 1 ;;
  esac
  ib_hca="=$(printf '%s' "$hcas" | sed 's/,/:1,/g'):1"
  role_work="$BASE_WORK/$slot"
  install -d -m 0700 "$role_work"
  cache_dir="$COMPILE_CACHE_ROOT/$slot"
  install -d -m 0700 "$cache_dir"
  write_mc_config "$role_work/mooncake.json" "$hcas"
  engine_id="$SERVICE_PREFIX-$slot-$NODE_ROLE-$(cat /proc/sys/kernel/random/uuid)"
  kv=$(cat <<EOF
{
  "kv_connector":"MultiConnector","kv_role":"kv_both","engine_id":"$engine_id",
  "kv_load_failure_policy":"fail",
  "kv_connector_extra_config":{"connectors":[
    {"kv_connector":"MooncakeConnector","kv_role":"$pd_role",
     "kv_connector_extra_config":{"mooncake_protocol":"rdma","device_name":"$hcas"}},
    {"kv_connector":"MooncakeStoreConnector","kv_role":"$store_role",
     "kv_connector_extra_config":{"cache_prefix":"$CACHE_NAMESPACE",
       "load_async":true,"lookup_async":true,"store_compact_group_payloads":true,
       "remove_superseded_checkpoints":$retire_checkpoints}}
  ]}
}
EOF
)
  docker run -d --pull=never --restart unless-stopped --name "$SERVICE_PREFIX-$slot-$NODE_ROLE" \
    --init --network host --shm-size 64g --memory 64g --memory-swap 64g \
    --gpus "$gpu_request" --device /dev/infiniband \
    --cap-add IPC_LOCK --cap-add SYS_NICE --ulimit memlock=-1:-1 \
    --ulimit stack=67108864:67108864 \
    -v "$MODEL_DIR:/model:ro" -v "$role_work:/work" \
    -v "$cache_dir:/root/.cache" \
    -v "$BASE_WORK/numa_wrapper.sh:/usr/local/lib/python3.12/dist-packages/vllm/utils/numa_wrapper.sh:ro" \
    -v "$TOKENIZER_DIR:/run/glm-tokenizer:rw" \
    -e VLLM_GLM53_TOKENIZER_SHARED_PATH=/run/glm-tokenizer/segments.db \
    -e VLLM_GLM53_TOKENIZER_SHARED_GIB="$shared_gib" \
    -e VLLM_GLM53_TOKENIZER_CACHE_GIB="$cache_gib" \
    -e VLLM_GLM53_TOKENIZER_LOCAL_L1_MIB=128 \
    -e VLLM_GLM53_TOKENIZER_CACHE_EPOCH_FILE=/run/glm-tokenizer/epoch \
    -e MC_TCP_BIND_ADDRESS="$NODE_IP" -e MOONCAKE_REQUESTER_LOCAL_HOSTNAME="$NODE_IP" \
    -e MC_ENABLE_DEST_DEVICE_AFFINITY=1 \
    -e WITH_NVIDIA_PEERMEM=0 \
    -e VLLM_HOST_IP="$NODE_IP" -e NCCL_SOCKET_IFNAME="$NET_IF" \
    -e GLOO_SOCKET_IFNAME="$NET_IF" -e NCCL_IB_HCA="$ib_hca" \
    -e NCCL_IB_GID_INDEX="$GID_INDEX" -e MC_GID_INDEX="$GID_INDEX" -e NCCL_DEBUG=INFO \
    -e MAX_JOBS=4 -e FLASHINFER_NVCC_THREADS=1 -e VLLM_ENGINE_READY_TIMEOUT_S=2400 \
    -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_SERVER_DEV_MODE=1 \
    -e VLLM_GLM53_NVFP4_FUSED_MOE=1 \
    -e VLLM_GLM53_NVFP4_STAGE_FC2_INPUT=1 \
    -e VLLM_GLM53_NVFP4_VECTOR_WEIGHTS=1 \
    -e MOONCAKE_CONFIG_PATH=/work/mooncake.json \
    -e VLLM_MOONCAKE_BOOTSTRAP_PORT="$bootstrap" -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120 \
    ${role_env[@]+"${role_env[@]}"} \
    --entrypoint vllm "$image" serve /model \
    --served-model-name "$MODEL_NAME" --host "$NODE_IP" --port "$port" \
    --api-key "$INTERNAL_API_KEY" --api-server-count "$apis" \
    --tensor-parallel-size 1 --data-parallel-size "$dp" --data-parallel-size-local "$dp" \
    --data-parallel-backend mp --data-parallel-rpc-port "$rpc" \
    --enable-expert-parallel --enable-ep-weight-filter \
    --max-model-len auto --max-num-seqs "$seq" --max-num-batched-tokens "$batch" \
    --gpu-memory-utilization "$gpu_util" --kv-cache-dtype nvfp4 \
    --block-size 64 --prefix-match-unit 256 --mamba-cache-mode align \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --kv-transfer-config "$kv" --scheduler-reserve-full-isl \
    --enable-prefix-caching --reasoning-parser glm47 \
    --enable-auto-tool-choice --tool-call-parser glm47 \
    --safetensors-load-strategy lazy --enable-cumem-allocator --seed 42 \
    --enable-prompt-tokens-details --enable-force-include-usage --numa-bind \
    ${graph_args[@]+"${graph_args[@]}"}
}
~~~

P 启动两个 Prefill 实例：

~~~bash
test "$NODE_ROLE" = p
start_model p0
start_model p1
~~~

D1 启动 Decode 实例：

~~~bash
test "$NODE_ROLE" = d1
start_model d0
~~~

D2 启动 Decode 实例：

~~~bash
test "$NODE_ROLE" = d2
start_model d0
~~~

四个模型入口都健康后分配 Store：

~~~bash
for endpoint in "$P0_URL" "$P1_URL" "$D1_URL" "$D2_URL"; do
  curl -fsS "$endpoint/health" || exit 1
done
~~~

### 4. Mooncake Store · 三台主机

三台各执行一次。L2 默认分别为 700、750、750 GiB；L3 挂在 NVMe 的 `$L3_MOUNT/l3/$SERVICE_PREFIX`，每台按需使用最多 2.5 TB。Store 容器保留 832 GiB 内存上限，覆盖 L2 和运行开销。

~~~bash
case "$NODE_ROLE" in
  p)  STORE_GIB="$L2_P_GIB" ;;
  d1) STORE_GIB="$L2_D1_GIB" ;;
  d2) STORE_GIB="$L2_D2_GIB" ;;
esac
if [[ ! "$STORE_GIB" =~ ^[1-9][0-9]{0,2}$ ]] || (( STORE_GIB > 750 )); then
  echo 'L2 容量必须为 1–750 GiB 的整数' >&2; exit 2;
fi
STORE_BYTES=$((STORE_GIB * 1024 * 1024 * 1024))
# 每台 NVMe 按需使用 2.5 TB（十进制）。
L3_MAX_BYTES=2500000000000
L3_DIR="$L3_MOUNT/l3/$SERVICE_PREFIX"
test "$(findmnt -n -T "$L3_MOUNT" -o TARGET)" = "$L3_MOUNT"
test ! -L "$L3_MOUNT/l3"
mkdir -p "$L3_MOUNT/l3"
test ! -L "$L3_DIR"
install -d -m 0700 "$L3_DIR"
test "$(realpath "$L3_DIR")" = "$L3_DIR"
test "$(findmnt -n -T "$L3_DIR" -o TARGET)" = "$L3_MOUNT"
L3_EXISTING_BYTES=$(du -sxB1 -- "$L3_DIR" | awk '{print $1}')
L3_FREE_BYTES=$(df -B1 --output=avail "$L3_DIR" | tail -1 | tr -d '[:space:]')
(( L3_FREE_BYTES + L3_EXISTING_BYTES >= L3_MAX_BYTES )) || {
  echo 'Store 需要 2.5 TB L3 容量。' >&2; exit 1;
}
LIBCUDA_PATH=$(readlink -f /lib/x86_64-linux-gnu/libcuda.so.1)
test -f "$LIBCUDA_PATH"
docker run -d --pull=never --restart unless-stopped --name "$SERVICE_PREFIX-store-$NODE_ROLE" \
  --init --network host --ipc host --runtime=runc \
  --memory 832g --memory-swap 832g --ulimit memlock=-1:-1 \
  --device /dev/infiniband --cap-add IPC_LOCK --cap-add SYS_NICE \
  -e NVIDIA_VISIBLE_DEVICES=void -e MC_GID_INDEX="$GID_INDEX" \
  -e MC_TCP_BIND_ADDRESS="$NODE_IP" -e MOONCAKE_REQUESTER_LOCAL_HOSTNAME="$NODE_IP" \
  -e NCCL_SOCKET_IFNAME="$NET_IF" -e GLOO_SOCKET_IFNAME="$NET_IF" \
  -v "$LIBCUDA_PATH:/opt/host-driver/libcuda.so.1:ro" \
  -e LD_LIBRARY_PATH=/opt/host-driver:/usr/local/cuda/lib64 \
  --mount "type=bind,src=$L3_DIR,dst=/mooncake-ssd" \
  -e MOONCAKE_OFFLOAD_FILE_STORAGE_PATH=/mooncake-ssd \
  -e MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR=bucket_storage_backend \
  -e MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY=lru \
  -e MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE="$L3_MAX_BYTES" \
  -e MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES="$L3_MAX_BYTES" \
  -e MOONCAKE_OFFLOAD_BUCKET_MAX_PHYSICAL_BYTES="$L3_MAX_BYTES" \
  -e MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=1342177280 \
  --entrypoint mooncake_client "$STORE_IMAGE" \
  --master_server_address="$MASTER_RPC" \
  --metadata_server="$MASTER_METADATA_URL" \
  --protocol=rdma --host="$NODE_IP" --port="$STORE_RPC_PORT" \
  --device_names="$NODE_HCAS" --global_segment_size="$STORE_BYTES" \
  --local_buffer_size=0 --enable_http_server=true --http_port="$STORE_HEALTH_PORT" \
  --enable_offload=true --start_offload_rpc_server=true --logtostderr=true
curl -fsS --retry 20 --retry-connrefused --retry-delay 3 "http://$NODE_IP:$STORE_HEALTH_PORT/health"
~~~

### 5. Router · P

三台 Store 健康后，在 P 核对两个实例的 bootstrap。每个端口应返回 4 个 DP，worker 地址应属于 `P_IP`。首次启动和 P 重启都按此检查。

~~~bash
test "$NODE_ROLE" = p
for port in "$P0_BOOTSTRAP_PORT" "$P1_BOOTSTRAP_PORT"; do
  curl -fsS "http://$P_IP:$port/query" | python3 -m json.tool
done
~~~

在 P 启动 Router。推理入口使用 `ROUTER_BASE_URL`，指标端口为 `ROUTER_METRICS_PORT`。`ROUTER_REQUEST_TIMEOUT_SECS` 默认 3600 秒，限制 Router 到任一上游的单次 HTTP 调用，包含响应体读取。

内网无鉴权是本部署的默认策略。需要客户端鉴权时，把 `ROUTER_CLIENT_VALIDATION_URLS` 设为可信身份服务的 HTTPS 校验地址（多个用逗号分隔），与 P/D 的内部 key 独立。客户端使用 `Authorization: Bearer <CLIENT_KEY>`；校验服务对有效 key 返回 200，对无效、过期或已撤销的 key 返回 401/403。Router 逐请求核验；没有有效凭证返回 401，身份服务不可用返回 503。启用后，模型列表、健康检查与推理请求均使用有效客户端凭证。

已有部署调整此值时，排空在途请求后用新参数重建 Router 容器，沿用原 IP、端口和镜像；`docker restart` 沿用旧参数。vLLM、Mooncake 和 KV 缓存保持原样。

~~~bash
docker run -d --pull=never --name "$SERVICE_PREFIX-router" \
  --runtime=runc --network host --restart unless-stopped \
  --memory 32g --memory-swap 32g --cpuset-mems 0 \
  --ipc private --shm-size 64m --ulimit nofile=1048576:1048576 \
  --log-driver local --log-opt max-size=128m --log-opt max-file=5 \
  -e OPENAI_API_KEY="$INTERNAL_API_KEY" -e NVIDIA_VISIBLE_DEVICES=void \
  -e NO_PROXY='*' -e no_proxy='*' \
  -e API_KEY_VALIDATION_URLS="${ROUTER_CLIENT_VALIDATION_URLS:-}" -e VLLM_ROUTER_GLM_INLINE_IMAGE_MAX_BYTES \
  -e VLLM_ROUTER_PD_REUSE_PROMPT_IDS=1 \
  -e VLLM_ROUTER_PD_PREFILL_FORCE_LENGTH_CAP=1 \
  --entrypoint /usr/local/bin/vllm-router "$ROUTER_IMAGE" \
  --host "$ROUTER_BIND_HOST" --port "$ROUTER_PORT" \
  --prometheus-host "$ROUTER_METRICS_BIND_HOST" --prometheus-port "$ROUTER_METRICS_PORT" \
  --vllm-pd-disaggregation --kv-connector mooncake \
  --prefill "$P0_URL" "$P0_BOOTSTRAP_PORT" --prefill "$P1_URL" "$P1_BOOTSTRAP_PORT" \
  --decode "$D1_URL" --decode "$D2_URL" \
  --prefill-policy cache_aware --decode-policy round_robin \
  --max-tree-size 2147483648 --eviction-interval 120 --max-concurrent-requests 4096 \
  --disable-retries --log-level warn \
  --request-timeout-secs "$ROUTER_REQUEST_TIMEOUT_SECS" --worker-startup-timeout-secs 7200
~~~

### 验收 · P

~~~bash
curl -fsS "$MASTER_METRICS_URL" | grep '^master_total_capacity_bytes '
router_headers=()
if [[ -n ${ROUTER_CLIENT_VALIDATION_URLS:-} ]]; then
  router_headers=(-H "Authorization: Bearer ${ROUTER_CLIENT_API_KEY:?填写健康检查凭证}")
fi
curl -fsS "${router_headers[@]}" "$ROUTER_URL/v1/models"
curl -fsS "${router_headers[@]}" "$ROUTER_URL/workers" |
  python3 -c 'import json,sys; w=json.load(sys.stdin)["workers"]; print(sum(x["is_healthy"] for x in w), "/", len(w))'
~~~

默认 L2 总量为 2200 GiB，Router 为 24/24 路健康。

推理请求直接向 Router 的 `/v1/chat/completions` 发送，模型填写 `glm-5.3-flash`。模型列表从 Router 的 `/v1/models` 获取，入口沿用推理请求的认证配置。Responses 的图片支持 URL 字符串和带 url 的对象；未指定 detail 时为 auto。

### 接口与验收范围

| 项目 | 当前结论 |
| --- | --- |
| 三协议核心功能 | 线上已验证 Chat、Responses、Messages 的 JSON/SSE、完整历史多轮、JSON Schema、内联 PNG、单工具调用和工具结果回填。未知模型返回 404，典型非法输入返回 400；Responses 输出预算用完发送 `response.incomplete`。Messages token 计数已通过。 |
| Chat 工具和计量 | GLM Chat 同时返回 `reasoning_content` 与兼容字段 `reasoning`，内容相同。新版引擎的 CPU 回归覆盖实际工具调用以 `finish_reason=tool_calls` 正常结束、预算截断仍为 `length`，以及最终 usage 只发一块；上线后复跑真实请求。账单按每个请求的最终累计 usage 记一次。 |
| 参数、工具及异常边界 | 待验收温度、top_p、惩罚项和 logit_bias 的默认值、上下界及实际效果，停止串范围，非法 Schema 及其流式组合，并行/禁用工具与错误/乱序 ID，实际权限、限流及客户端断开取消。CPU 回归已验证可选鉴权按请求检查（无效或撤销返回 401，身份服务不可用返回 503），以及 Responses 的 completed、incomplete 和 failed 流式终态。 |
| 多模态及复杂历史 | 内联图片、多轮图和超过 4 MiB 的 Responses 样本已验证；CPU 已验证 HTTP URL 图片解码、损坏图片拒绝及三协议视频输入；多图、更多格式尺寸及复杂历史的覆盖范围以实测记录为准。视频端到端画面理解须用新镜像实际推理复验；CPU 的协议转换和媒体预处理验收与模型回答质量分别记录。Responses 不存储会话，续聊使用完整历史。 |
| Token 与缓存 | 线上已验证三协议缓存字段和 L2 跨 P 读回；三台 L3 隔离池 NVMe 冷读逐字节一致。GLM 原厂同版本的逐项 Token 对照、长前缀命中率、4 小时缓存 TTL 和生产请求实际 L3 命中待取证。 |
| 超时、性能及交付 | 1P2D 部署默认 Router 上游单次 HTTP 调用总超时为 3600 秒，包含响应体读取。持续出流不会重置计时。网关和客户端超时需配齐；1 小时端到端长任务未验收。正式分长度 TTFT/TPOT、效果评测使用 GLM 自身基线；额度和技术支持按交付约定另验。 |

本部署保留模型原生思考，不支持联网搜索。DeepSeek 规范仅用于对照验收方法；未经实测的项目不计通过。

## 服务重启后的检查

Master、Store、vLLM 和 Router 使用 Docker 的 `unless-stopped` 策略：进程退出时自动重新启动，手动停止的容器保持停机。整机重启后，按 Master → 模型 → Store → Router 核对就绪状态。

停止放流，等在途请求结束。重启后做一条经过 Router 的真实 P→D 请求，看到生成结果再恢复流量。

| 重启对象 | 核对与恢复 |
| --- | --- |
| Router | 核对两个 `--prefill` 的 HTTP 和 bootstrap 端口、两个 `--decode` 的 HTTP 地址。`OPENAI_API_KEY` 与 P/D 内部 key 一致。确认 `/workers` 为 24/24，再发一条实际推理请求。 |
| vLLM P | 同一实例沿用原 HTTP、bootstrap 地址及同 GPU 槽位的编译缓存。`/health` 就绪且 `/query` 登记齐 4 个 DP 后发送真实请求。相同 engine ID 下，D 在旧 worker 端口失效时重新查目录；若重建时更换了 engine ID，核对 Router 注册并按需刷新。 |
| vLLM D1 / D2 | 复用本机 d0 编译缓存，核对 D 的 IP、HTTP、GPU 槽位和 NVFP4 格式。等 /health 和 Router 对应 D 的 8 个 DP 恢复。D 地址或端口变化时，更新 --decode 后重建 Router。 |
| Mooncake Master 或 Store | Master 先就绪，再启动或重连各 Store。核对 Master RPC、metadata、Store RPC 与健康端口，以及 RDMA 地址和 L3 挂载目录。Master 指标显示三台 L2 共 2200 GiB 后做一次新 KV 的写入和读回；旧缓存按冷缓存处理。若 Master 地址变更或 vLLM 客户端未恢复，更新各实例 work/mooncake.json 后重启受影响的 vLLM；涉及 P 时最后刷新 Router。 |

地址或端口变更时，同步三台的 `deploy.env` 后重建受影响的容器。客户端入口变化时，更新 `ROUTER_BASE_URL`。

## 发送推理请求

Responses 的 `tools` 同时接受扁平函数定义和 `{"type":"function","function":{...}}` 包装。Router 转发前展开 `name`、`description`、`parameters`、`strict`；同名字段值冲突时返回 HTTP 400（`invalid_function_tool`）。续聊仍在 `input` 中携带完整历史。

客户端统一访问 P 节点上的 Router，`ROUTER_BASE_URL` 是客户端可达地址，含 `/v1`。模型名为 `glm-5.3-flash`，推理请求默认无需 API key。以下是流式请求，示例使用 `max` 思考（也可填 `high`），输出上限为 4096 token（含思考）。

~~~bash
# 独立客户端填写实际地址。部署会话中已设置 ROUTER_BASE_URL，可跳过这一行。
ROUTER_BASE_URL='http://<ROUTER_HOST>:<ROUTER_PORT>/v1'
curl -fsS -N "${ROUTER_BASE_URL%/}/chat/completions" \
  -H 'Content-Type: application/json' \
  --data-binary @- <<'JSON'
{
  "model": "glm-5.3-flash",
  "messages": [{"role": "user", "content": "请用三句话介绍前缀缓存。"}],
  "reasoning_effort": "max",
  "max_tokens": 4096,
  "stream": true,
  "stream_options": {"include_usage": true}
}
JSON
~~~

流正常结束时返回 `data: [DONE]`；整包 JSON 使用 `"stream": false`。经 SSH 转发时，`ROUTER_BASE_URL` 填 `http://127.0.0.1:<LOCAL_FORWARD_PORT>/v1`。

收到实际工具调用且生成正常结束时，流式和非流式 Chat 都返回 `finish_reason=tool_calls`；输出上限导致的截断仍返回 `length`。流式 Chat 的最终计量位于 `[DONE]` 前 `choices: []` 的 usage 块。仅在 `stream_options` 同时启用 `include_usage` 与 `continuous_usage_stats` 时才发过程累计值；过程值供展示，账单采用最终块。

图片请求使用同一接口。将消息的 `content` 改成数组，其中问题用 `text`，每张图用一个 `image_url`，例如：

~~~json
[
  {"type": "text", "text": "请描述这些图片。"},
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64_IMAGE_1>"}},
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64_IMAGE_2>"}}
]
~~~
