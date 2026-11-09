# GLM5.3 Flash 1103 · 1P2D 部署

1P2D按主机计数：1台P主机、2台D主机。P主机分成两个EP4实例，D主机各运行一个EP8实例，共24张GPU、4个模型HTTP入口。示例为P82 / D83、D84；可通过下文变量替换为另一组主机，网卡按实际拓扑设置。例如使用P85 / D86、D88时，在三台主机均设置：

~~~bash
export P_IP=11.8.0.85 D1_IP=11.8.0.86 D2_IP=11.8.0.88
export MASTER_IP=11.8.0.86 ROUTER_IP=11.8.0.86
# 第3节的NODE_ID分别填写85、86、88，NODE_IP使用对应主机地址。
~~~

## 启动字段说明与使用示例

引擎中的五个 GLM 优化开关默认均为 `0`，在容器创建前显式设置。P 的基础模型专家保留在 GPU，D 保留完整 MTP3；这些字段不切换注意力后端，也不关闭异步调度、CUDA graph 或通信计算重叠。

| 字段 | P | D | 作用与使用案例 |
| --- | --- | --- | --- |
| `VLLM_GLM53_NVFP4_FUSED_MOE` | 1 | 1 | 启用 SM120 GLM NVFP4 MoE 融合；用于 EP4 P / EP8 D，超过融合分发范围的 batch 继续用原生后端。 |
| `VLLM_GLM53_NVFP4_STAGE_FC2_INPUT` | 1 | 1 | 在融合核中暂存 FC2 输入，减少重复读取；与上面的融合开关配套使用。 |
| `VLLM_GLM53_NVFP4_VECTOR_WEIGHTS` | 1 | 1 | 使用向量化权重读取；适用于本分支的 NVFP4 原生权重布局。 |
| `VLLM_GLM53_NVFP4_EXTENDED_PREFETCH` | 1 | 0 | 给 EP4 P 的每 rank 72 个专家扩展预取；EP8 D 不开启。 |
| `VLLM_GLM53_P_MTP_CONTEXT_ONLY` | 1 | 0 | 专用 P 只加载 MTP 注意力和输入融合必需的18个源键，完成 MTP KV 交接；D 或耦合实例设0，保留完整草稿生成。 |
| `VLLM_EXPERIMENTAL_HOT_TAIL_ADMISSION` | 1 | 0 | 为 P 异步读回后的末段 CoW 状态块留余量，减少抢占重算；诊断开关保持0。 |

下面是 Docker 参数片段；后文的完整命令已包含这些值：

~~~bash
# P / D 公共参数
-e VLLM_GLM53_NVFP4_FUSED_MOE=1 \
-e VLLM_GLM53_NVFP4_STAGE_FC2_INPUT=1 \
-e VLLM_GLM53_NVFP4_VECTOR_WEIGHTS=1
# 专用 EP4 P 额外配置；D 对应两项都设0
-e VLLM_GLM53_NVFP4_EXTENDED_PREFETCH=1 \
-e VLLM_GLM53_P_MTP_CONTEXT_ONLY=1
~~~

| Router 字段 | 使用值 | 作用与使用案例 |
| --- | --- | --- |
| `--prefill-policy` / `--decode-policy` | `cache_aware` / `round_robin` | P 按前缀复用选择，D 在健康 worker 间轮询；只调整后文登记的 P/D 地址即可用于不同主机数量。 |
| `VLLM_ROUTER_PD_REUSE_PROMPT_IDS` | 1 | D 复用 P 返回的精确 token IDs，避免同一长输入重复分词；两端 tokenizer 与 chat template 必须一致。 |
| `VLLM_ROUTER_PD_PREFILL_FORCE_LENGTH_CAP` | 1 | P 的单 token 请求忽略 EOS/stop，确保 Mooncake 发布可读取的 KV；D 保留客户端的输出长度和停止条件。 |
| `--max-tree-size` / `--eviction-interval` | 2147483648 / 120秒 | 前缀树每 worker/tenant 的字符预算及 LRU 清理周期；长期服务需要更小内存占用时，降低前者或缩短后者。 |
| `--max-concurrent-requests` | 4096 | Router 总在途请求上限，不是分词槽位；压测并发由客户端单独设置。 |

Router 使用从同名1103分支构建的 `router:glm53flash-1103`。构建命令见 Router 仓库 `DEPLOYMENT.md`；不加载估算分词器或 token 容量配置，也不分配估算分词槽位。Program scheduling 保持关闭。

## 1. 拓扑与固定参数

| 主机 | 实例 | GPU / DEP / EP | API / bootstrap / DP RPC 端口 |
| --- | --- | --- | --- |
| P82 | p0、p1 | GPU0–3 / GPU4–7；每实例TP1 / DP4 / EP4 | p0：18483 / 19083 / 19550；p1：18493 / 19093 / 19650 |
| D83、D84 | 每台d0 | GPU0–7；每实例TP1 / DP8 / EP8 | 18483 / 19083 / 19550 |
| 三台模型主机 | 每台一个Store | 参考750GiB/台，共2250GiB | RPC61244 / HTTP61245 |
| D83 | Master、Router | Router无CPU quota；64GiB RAM / 96GiB RAM+swap、mems0 | Master61241–61243；Router18300 / 18301 |

| 项目 | P | D |
| --- | --- | --- |
| 每 DP 的 seq / batch tokens | 24 / 2048 | 16 / 1024 |
| CUDA graph | 原生 native auto，实际解析上限 192 | FULL_DECODE_ONLY，上限 64 |
| API 进程 | 4/实例，即 8/台 | 1/实例，即 1/台 |
| Tokenizer 共享池 | 15.5 GiB/台，p0/p1 共用 | 15.9375 GiB/台 |
| Tokenizer 私有 L1 | 64 MiB/API，共 0.5 GiB/台 | 64 MiB/API，共 0.0625 GiB/台 |
| Tokenizer 总预算 | 16 GiB/台 = 2 GiB/GPU，已含 L1 | 16 GiB/台 = 2 GiB/GPU，已含 L1 |
| 目的 HCA 亲和 | `MC_ENABLE_DEST_DEVICE_AFFINITY=1` | `MC_ENABLE_DEST_DEVICE_AFFINITY=1` |
| 末段准入预算（P 抢占优化） | 部署显式设 1；引擎代码默认仍为 0 | 沿用代码默认 0 |
| KV 接收线程 | `VLLM_MOONCAKE_LOAD_RECV_THREADS=4` | 沿用原生配置 |

两端保持原生 async、FP8 KV、MTP3、fullISL、scheduler 水位 0、TP1、GPU utilization 0.95、prefix match unit 256、Mamba align、原生 allgather_reducescatter。CLI block size 为 64，GLM 混合缓存实际解析物理页为 8960；max model len 使用 auto，启动后核对 1048576 上下文。P 保留原生图选择，启动后的上限应为 192。

NVFP4 融合、FC2 输入 staging 和向量化权重读取在 P/D 均开启；P 额外开启扩展预取和 MTP context-only。融合仅用于 GLM SM120 / EP4、EP8：P 的 dense batch 上限 192，稀疏 batch 最大 256 且总 token 不超过 512；D 上限 128。其余形状走原生后端。共享专家、注意力后端、异步调度与计算通信重叠保持原生路径。

P 保留基础模型全部专家及 MTP 注意力、融合所需的 18 个源权重，不加载只用于草稿生成的 MTP FFN/专家和 head；D 保留完整 MTP3。该模式只适用于专用 EP4 P，不用于 D 或耦合实例。P 完成事件在 MTP KV 写入之后发布，D 使用完整状态继续投机解码。

Tokenizer 的 `CACHE_GIB` 参数 P=2、D=16 是兼容参数；共享模式的整机预算以共享池加全部 API 的 L1 计算，不能再额外加一份 16 GiB。各模型主机应有足够 tmpfs 和内存；Store 新分配前参考可用内存大于 850 GiB，全部服务启动后宿主仍至少保留 16 GiB 可用内存。

## 2. 源码与镜像

使用本分支源码和 `integrations/glm53flash/Dockerfile` 构建引擎。P/D 使用同一份 GLM 源码，优化开关在镜像中默认关闭，由第 6 节按角色启用。`integrations/glm53flash/release.json` 校验原生基底、覆盖文件和运行源码。

| 组件 | 镜像要求 |
| --- | --- |
| P / D 引擎 | 匹配清单的 GLM 原生基底，包含 CUDA 编译器、FlashInfer 0.7.1 与自动调优补丁；构建后设置 `P_IMAGE`、`D_IMAGE`。 |
| Store | 配套 Mooncake SDK；本地参考 ID：`sha256:bd6c77d709b8225ebf7b4a9a3890509a4e2df87af846fcc7230b65a7a57d16f7`。 |
| Master | `Dockerfile.mooncake-master` 构建，包含实际占用计账和大块空间回收。 |
| Router | `router:glm53flash-1103`，从 Router 同名分支构建，分发后核对实际 image ID。 |

NVFP4 内核在实例初始化时编译为 SM120a。保留挂载的 `/root/.cache` 以复用匹配的编译和调优缓存，不在请求路径编译。P2P 内核使用 FlashInfer 的 JIT 和 FFI 头文件，无需 Cake 后端。不要通过 `FLASHINFER_EXTRA_CUDAFLAGS` 或 NVCC prepend/append flags 覆盖架构目标。

### 在构建机生成镜像

在仓库根目录执行。构建机应已导入匹配的 GLM 基底，例如 `local/glm53flash-1103:router-policy-fi071-20261107`；它是本地镜像名，不是公开 registry 地址。本地 image ID 不是 registry manifest digest。分发后在各节点核对 ID，`--pull=never` 要求镜像预先就绪。

~~~bash
: "${ENGINE_BASE_IMAGE:?填写匹配的 GLM 原生基底}"
: "${ENGINE_IMAGE:?填写引擎目标 tag}"
docker build -f integrations/glm53flash/Dockerfile \
  --build-arg BASE_IMAGE="$ENGINE_BASE_IMAGE" -t "$ENGINE_IMAGE" .
export P_IMAGE="$ENGINE_IMAGE" D_IMAGE="$ENGINE_IMAGE"

: "${MOONCAKE_SOURCE:?填写本地 Mooncake 源码绝对路径}"
: "${MASTER_IMAGE:?填写 Master 目标 tag}"
: "${MASTER_BASE_IMAGE:?填写匹配 Store SDK 的基底}"
test "$(git -C "$MOONCAKE_SOURCE" rev-parse HEAD)" = \
  "$(jq -er '.commit' integrations/glm53flash/mooncake-master.json)"
bash integrations/glm53flash/build-mooncake-master.sh \
  "$MOONCAKE_SOURCE" "${MOONCAKE_BUILD_DIR:-/tmp/mooncake-1103-build}" \
  "$PWD/mooncake-master-runtime"
docker build -f integrations/glm53flash/Dockerfile.mooncake-master \
  --build-arg BASE_IMAGE="$MASTER_BASE_IMAGE" -t "$MASTER_IMAGE" .
~~~

Master 使用 Mooncake `719735896c86b56fabec6cf3e825fb2ea640597a`；构建脚本参数为源码目录、构建目录和 runtime 输出目录。输出应位于构建上下文的 `mooncake-master-runtime/`。Router 使用独立镜像，P `cache_aware`、D `round_robin`，启用 prompt ID 复用和 length-capped 交接，不执行 token 容量估算或分词槽位调度；program scheduling 保持关闭。

## 3. 每台主机的公共准备

需要 Docker、NVIDIA Container Toolkit、RDMA 驱动，以及 Bash、curl、jq。以下会话变量需在各目标主机设置；`NODE_ID` 分别填P与两个D的主机号，服务 IP、网卡和路径按实际环境覆盖。P/D/Router 使用同一份 API key，通过交互输入或受控环境注入，不写入文档或版本库。

所有模型节点的 `MODEL_DIR` 应指向同一版本模型文件，包括 tokenizer 配置、tokenizer.json 和 chat template；P/D 的渲染和分词语义保持一致。本方案 Router 不加载估算 Tokenizer，不挂载模型目录或估算容量配置；D 通过 prompt ID 复用直接使用 P 的分词结果。

~~~bash
test "$(id -u)" -eq 0
set -euo pipefail
umask 077
export NODE_ID=82  # 分别在P82、D83、D84执行；主机号可按实际环境调整
case "$NODE_ID" in 81|82|83|84|85|86|87|88) ;; *) exit 1 ;; esac
export NODE_IP="${NODE_IP:-11.8.0.$NODE_ID}"
export MASTER_IP="${MASTER_IP:-11.8.0.83}"
export ROUTER_IP="${ROUTER_IP:-11.8.0.83}"
export P_IP="${P_IP:-11.8.0.82}"
export D1_IP="${D1_IP:-11.8.0.83}" D2_IP="${D2_IP:-11.8.0.84}"
export P_HOSTS="$P_IP"
export D_HOSTS="$D1_IP $D2_IP"
case "$NODE_IP" in
  "$P_IP"|"$D1_IP"|"$D2_IP") ;;
  *) echo 'NODE_IP不在本次1P2D主机列表中' >&2; exit 1 ;;
esac
export MODEL_DIR="${MODEL_DIR:-/data/models/GLM-5.3-Flash-NVFP4}"
export MODEL_NAME="${MODEL_NAME:-glm-5.3-flash-nvfp4}"
export SERVICE_PREFIX="${SERVICE_PREFIX:-glm53flash-1103-1p2d}"
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

## 5. Store：三台模型主机各执行一次

每台机器一个共享 Store，不是每 GPU 一份。参考750GiB/台；按本机服务预热后的余量配置。客户端的 global segment 为 0，只注册 2 GiB 本地传输缓冲；Store 的 local buffer 为 0。

~~~bash
: "${STORE_IMAGE:?设置已导入的 Store 镜像}"
STORE_GIB="${STORE_GIB:-750}"
STORE_BYTES=$((STORE_GIB * 1024 * 1024 * 1024))
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
  --device_names="$HCAS_ALL" --global_segment_size="$STORE_BYTES" \
  --local_buffer_size=0 --enable_http_server=true --http_port=61245 \
  --enable_offload=false --logtostderr=true
curl -fsS "http://$NODE_IP:61245/health"
~~~

### 5.1. L2 容量建议：服务预热后剩余内存的 85%

本文的 KV 缓存层级是：L1 为 GPU KV，L2 为 Mooncake Store 的主机 DRAM，L3 为可选 SSD；与 Tokenizer 的 64 MiB 私有 L1 是两套缓存。上述750GiB是参考容量，按实际余量设置 STORE_GIB。

**建议模型、Router、共享分词等服务启动并预热后，将剩余可用内存的 85% 用于 L2，留下 15% 给运行波动和额外开销。** 各主机分别计算；Router 所在主机也要计入其内存使用。读取 `MemAvailable`，不用 `MemFree`。

如果已经启动 L2，应先加回当前 L2 实际划出的池容量，再算新的总预算；不要在原池之外再额外分配剩余内存的 85%。以下只计算，不调整运行中的服务：

~~~bash
# 尚未划 L2 池时填 0；若当前就是上面的 750 GiB 池，填 805306368000。
: "${L2_CURRENT_BYTES:?填写当前已划出的 L2 池字节数，未划池填 0}"
MEM_AVAILABLE_KIB=$(awk '$1 == "MemAvailable:" {print $2}' /proc/meminfo)
L2_BUDGET_BYTES=$(( (MEM_AVAILABLE_KIB * 1024 + L2_CURRENT_BYTES) * 85 / 100 ))
printf '建议 L2 总池容量：%s bytes\n' "$L2_BUDGET_BYTES"
~~~

采用该预算时，将 Store 命令中的 `--global_segment_size="$STORE_BYTES"` 替换为本机计算值，同时调整 Store 容器的 `--memory` / `--memory-swap`，给池之外的 SDK、RDMA 注册元数据及可选 SSD staging 留余量，宿主仍至少保留 16 GiB 可用内存。已有池的扩缩容在停用相关请求后重建 Store，不清理正在使用的池。第 8 节的 Master 总容量检查也改为各机 L2 池容量之和，以实际总容量为准。

L2 使用配套 Master 的高水位淘汰与大块空间回收；本方案保留 `eviction_high_watermark_ratio=0.98` 及 `MC_EVICTION_FIT_*` 参数。85% 是主机内存分配建议，0.98 是已分配池内的淘汰水位，两者分别配置。

### 5.2. L3 SSD 缓存（可选，默认关闭）

L3 由 Mooncake Store 的 SSD offload 管理，Router 不管理 KV 落盘。Master、Store 和模型默认关闭 L3。使用配套 SSD offload 接口时，按下面配置，Router 保持轮询。

1. 每台 Store 主机准备独立的本地 NVMe 目录，并按实际磁盘容量设置字节配额，避开模型、测试 DB 和日志目录。
2. Master 启动参数增加 `--enable_offload=true --offload_on_evict=true`，使 DRAM 淘汰选中的对象进入 SSD；保留现有 Master 补丁和其他启动参数。
3. Store 的 `docker run` 在镜像名前增加下列目录挂载与环境变量，并把 `mooncake_client` 的 `--enable_offload=false` **替换**为 `--enable_offload=true`：

~~~bash
export L3_DIR="${L3_DIR:-/nvme/mooncake-ssd}"
: "${L3_MAX_BYTES:?按本机 SSD 余量填写 L3 配额，单位 bytes}"
install -d -m 0700 "$L3_DIR"

# 追加到第 5 节 docker run 的镜像名前，不是单独执行的命令。
L3_DOCKER_ARGS=(
  -v "$L3_DIR:/mooncake-ssd"
  -e MOONCAKE_OFFLOAD_FILE_STORAGE_PATH=/mooncake-ssd
  -e MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR=bucket_storage_backend
  -e MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY=lru
  -e MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE="$L3_MAX_BYTES"
  -e MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES="$L3_MAX_BYTES"
)
# 在原命令中、--entrypoint 之前展开：
#   "${L3_DOCKER_ARGS[@]}"
# 原 mooncake_client 参数改为：--enable_offload=true
~~~

**模型侧：** P/D 使用的 Mooncake JSON 中将 `enable_offload` 设为 `true`，启用连接器的 SSD 读回分块预算；继续使用完整 KV group、compact payload 和原生异步加载。SDK 需与 Store 的 `LOCAL_DISK` 副本发现及 SSD 读取接口配套。

**验收：** 启用后验证 DRAM 未命中时出现 SSD 命中和真实磁盘读写，检查恢复输出与磁盘配额，再单独记录该方案的 TTFT/TPOT。SSD 缓存目录通过 `MOONCAKE_OFFLOAD_FILE_STORAGE_PATH` 配置，不使用 Master 的旧 `--root_fs_dir` 参数。

参数依据 Mooncake 官方 [SSD offload 部署文档](https://kvcache-ai.github.io/Mooncake/deployment/ssd/ssd-offload.html)；L3 需要配套版本支持，此部署默认关闭 L3。

## 6. 模型实例：P主机启动p0/p1，两个D主机各启动d0

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
      role_env=(-e VLLM_GLM53_P_MTP_CONTEXT_ONLY=1
                -e VLLM_GLM53_NVFP4_EXTENDED_PREFETCH=1
                -e VLLM_MOONCAKE_LOAD_RECV_THREADS=4
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
      image="${D_IMAGE:?设置已导入的D镜像}"
      dp=8; apis=1; seq=16; batch=1024; shared_gib=15.9375; cache_gib=16
      role_env=(-e VLLM_GLM53_P_MTP_CONTEXT_ONLY=0
                -e VLLM_GLM53_NVFP4_EXTENDED_PREFETCH=0)
      pd_role=kv_consumer; store_role=kv_consumer
      gpu_request='"device=0,1,2,3,4,5,6,7"'; hcas="$HCAS_ALL"
      port=18483; bootstrap=19083; rpc=19550
      graph_args=(--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":64}')
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
    -e VLLM_GLM53_NVFP4_FUSED_MOE=1 \
    -e VLLM_GLM53_NVFP4_STAGE_FC2_INPUT=1 \
    -e VLLM_GLM53_NVFP4_VECTOR_WEIGHTS=1 \
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

if [ "$NODE_IP" = "$P_IP" ]; then
  start_model p0
  start_model p1
else
  start_model d0
fi
~~~

末段准入预算用于异步加载后需要额外 CoW 状态块的特定 producer 请求，保护余量以减少抢占重算。引擎 `VLLM_EXPERIMENTAL_HOT_TAIL_ADMISSION` 默认 0，本部署只在 P 的命令里设为 1；诊断开关 `VLLM_HOT_TAIL_ADMISSION_DIAGNOSTICS` 保持 0。FP8、MTP3、原生异步、图选择和 fullISL 与该开关分别配置。

## 7. Router：在 Router 主机执行

等待全部4个模型实例完成启动，登记2个EP4 P和2个EP8 D，直接连接原生 vLLM HTTP API。P 使用 `cache_aware`，D 使用 `round_robin`。Program scheduling 保持关闭。`--max-concurrent-requests 4096` 是 Router 总请求并发上限，不是分词槽位数。

Router 使用 `router:glm53flash-1103`，不设 CPU quota；内存限制 64 GiB，RAM+swap 总上限 96 GiB，`cpuset-mems=0`。保留 prompt ID 复用、P 单 token length-capped 交接和 listener backlog 4096；模型共享分词缓存与 Mooncake 缓存由 P/D 管理。镜像已包含 `/usr/local/lib/router-listener.so`。

~~~bash
if [ "$NODE_IP" = "$ROUTER_IP" ]; then
  export ROUTER_IMAGE="${ROUTER_IMAGE:-router:glm53flash-1103}"
  read -r -a p_hosts <<< "$P_HOSTS"
  read -r -a d_hosts <<< "$D_HOSTS"
  test "${#p_hosts[@]}" -eq 1
  test "${#d_hosts[@]}" -eq 2
  routes=()
  for ip in "${p_hosts[@]}"; do
    routes+=(--prefill "http://$ip:18483" 19083 --prefill "http://$ip:18493" 19093)
  done
  for ip in "${d_hosts[@]}"; do
    routes+=(--decode "http://$ip:18483")
  done

  docker run -d --pull=never --name "$SERVICE_PREFIX-router" \
    --runtime=runc --network host --restart no \
    --memory 64g --memory-swap 96g --cpuset-mems 0 \
    --ipc private --shm-size 64m --ulimit nofile=1048576:1048576 \
    --log-driver local --log-opt max-size=128m --log-opt max-file=5 \
    -e OPENAI_API_KEY="$API_KEY" -e NVIDIA_VISIBLE_DEVICES=void \
    -e NO_PROXY='*' -e no_proxy='*' \
    -e VLLM_ROUTER_PD_REUSE_PROMPT_IDS=1 \
    -e VLLM_ROUTER_PD_PREFILL_FORCE_LENGTH_CAP=1 \
    -e LD_PRELOAD=/usr/local/lib/router-listener.so \
    -e AI_INFRA_LISTEN_BACKLOG=4096 \
    --entrypoint /usr/local/bin/vllm-router "$ROUTER_IMAGE" \
    --host 0.0.0.0 --port 18300 \
    --prometheus-host 0.0.0.0 --prometheus-port 18301 \
    --vllm-pd-disaggregation --kv-connector mooncake \
    "${routes[@]}" \
    --prefill-policy cache_aware --decode-policy round_robin \
    --max-tree-size 2147483648 --eviction-interval 120 --max-concurrent-requests 4096 \
    --disable-retries --api-key "$API_KEY" --log-level warn \
    --request-timeout-secs 600 --worker-startup-timeout-secs 7200
fi
~~~

### Router 自身的缓存淘汰

`cache_aware` 的前缀树使用后台 LRU，`--eviction-interval` 默认 120 秒；`--max-tree-size 2147483648` 按每个 worker/tenant 的保留字符数计量，不是整个 Router 的 RAM 字节上限。超过阈值的旧前缀在下一轮淘汰，不会把所有请求历史永久保留。

64 GiB 是容器硬内存上限，不是按 RSS 到达该值就自动淘汰的目标。树节点开销、不同 worker 的条目、在途请求和两个淘汰周期之间的新增内容仍占内存，因此仍可能触发 OOM。持续运行观察实际 RSS 和峰值，需要控制占用时调低 `--max-tree-size` 或缩短 `--eviction-interval`，另行对照缓存命中与性能；可按缓存命中和内存峰值调整上述参数。

## 8. 启动核验

启动顺序为 Master → 全部 Store → D / P 模型实例 → Router。保留模型启动日志，确认 P 原生图上限 192、D 图上限 64；原生 FlashInfer 预热和匹配的调优缓存复用保持生效。

在 Router 主机运行下列只读检查。模型预热期间可待其就绪后重试；24个健康worker是24个DP rank，对应4个独立HTTP origin。

~~~bash
for ip in $P_HOSTS $D_HOSTS; do
  curl -fsS "http://$ip:61245/health"
done
curl -fsS "http://$MASTER_IP:61243/metrics" |
  awk '$1 == "master_total_capacity_bytes" {print; ok=($2 == 2415919104000)} END {exit !ok}'

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
  check_model "http://$ip:18483" 8 16 1024 64
done
curl -fsS -H "Authorization: Bearer $API_KEY" "http://$ROUTER_IP:18300/workers" |
  jq -e '.total == 24 and (.workers | length == 24) and all(.workers[]; .is_healthy == true)'
~~~

各模型主机再确认宿主 `MemAvailable` 至少 16 GiB，Tokenizer 目录/epoch 的 uid、gid 为 0，p0/p1 的挂载指向同一共享池。P 启动日志应有每个 DP scheduler 的 `P_HOT_TAIL_BUDGET init`，4 条/实例，`enabled=True`、`progress_enabled=False`。

Router 核对镜像 ID、D 侧轮询、复用及 length-cap 开关、无 CPU quota、64 GiB / 96 GiB 内存配置。命令中不启用 `--enable-program-scheduling`。模型共享分词缓存继续由 P/D 管理，与Router请求路由独立。

模型日志应出现 `GLM NVFP4 fused MoE bound`，各 rank 的基础模型 42 个 MoE 层均完成绑定。FC2 staging 与 vector weights 为 True，P extended prefetch 为 True，D 为 False。真实请求应有融合分发记录，较大 batch 使用原生后端。核对 P 的 18 个必需 MTP 源键已加载，D 的完整 MTP3、上下文 1048576 和异步调度均生效。
