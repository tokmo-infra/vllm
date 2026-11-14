# 当前实跑部署命令

本轮 `run-d937e6f2d2b3`，B30092/93，P8+D8，C288/A576，P本地生成/PD在途5:9，输出上限2048。核对时间：2026-11-14T21:10:16.120564+08:00（北京时间）。本链路13个常驻容器的命令全部与只读Docker快照核对，519项检查通过；完整参数、镜像ID、挂载及脚本SHA256见[deployment-services.json](deployment-services.json)。此次补齐文件，没有执行启动、重启、清缓存或推理操作。

请求入口是 `92:19511`：按在途5:9分到P本地完整生成，或转给 `92:19500` 的原生PD Router，再执行P预填充→RDMA传KV→D解码。P的历史KV由Mooncake Store管理；D的HiSparse主机源池1024GiB独立于共享Store L2。

| 组件 | 节点 / 端口 | 当前镜像ID（缩写） | 完整启动脚本 |
| --- | --- | --- | --- |
| P：Prefill + 本地Decode | 92 / API19580；bootstrap配置29600 | `7499b963a8ad` | [prefill-92.sh](prefill-92.sh) |
| D：纯Decode + HiSparse | 93 / API19581；bootstrap配置29700 | `7499b963a8ad` | [decode-93.sh](decode-93.sh) |
| 混合Router：P本地/PD按在途5:9 | 92 / 19511 | `dc18c152f931`，CPU运行 | [router.sh](router.sh) |
| 原生PD Router：P→D、复用P的token IDs | 92 / 19500；指标127.0.0.1:20505 | `55b0495c0784` | [router-pd.sh](router-pd.sh) |
| Mooncake Store Master | 92 / RPC51561、HTTP metadata19590；指标127.0.0.1:20503 | `5a259407d16d` | [mooncake-master-92.sh](mooncake-master-92.sh) |
| Mooncake Store owner ×4 | 92 / transfer62040–62043、RPC62140–62143、HTTP62240–62243 | `a2071779007d` | [store/](store/) 中92的4个脚本 |
| Mooncake Store owner ×4 | 93 / 同一组端口，绑定本机RDMA地址 | `27ed3f34` | [store/](store/) 中93的4个脚本 |

P/D引擎共用一个镜像。混合Router复用旧P镜像作为CPU Python环境，设置 `NVIDIA_VISIBLE_DEVICES=void`，不加载模型；它与原生Rust PD Router是两个独立进程。所有脚本内均固定完整镜像ID，默认 `--dry-run`，`--execute` 才启动，并拒绝覆盖同名容器。

## Router实际使用的代码与挂载

混合Router源码见 [Router分支中的hybrid_router.py](https://github.com/tokmo-infra/router/blob/xinhao-vllm-router/tokmo/kimi31-hisparse-hybrid/hybrid_router.py)，本地notes导出目录也保留同SHA的源码副本；依赖见 [requirements.txt](requirements.txt)，镜像已包含aiohttp。Git归属为 `tokmo-infra/router` 的 `xinhao-vllm-router` 分支，混合Router提交 `4e4db92997074544577a33b97b579e535fd2a04b`；原生Router镜像标记的提交为 `96885611f4f7ab33d2661cb8aeffbabff44784b0`，原生可执行文件SHA256为 `28b5ff2e37ce85761a31092673f36200017c2fb21c363087805ff032af794cde`。

当前混合容器的只读挂载仍是92上的以下目录，映射为 `/hybrid`：

```text
/data2/bench/eng-xinhao/vllm-pd/kimi3.1/b300-92-93-p8-d8-tp8-dcp8-ep8/hisparse-hybrid-5to9
```

`router.sh` 默认保留这个实跑挂载位置。若改为新目录部署，先把Router分支中的 `hybrid_router.py` 放入该目录，再明确指定 `HYBRID_SOURCE_DIR="$ROOT"`；本地notes导出目录已包含同SHA副本。这只改变新容器的源码挂载位置；本次没有改正在运行的挂载。

原生PD Router参数为 `--prefill-policy cache_aware --decode-policy round_robin`，启用 `VLLM_ROUTER_PD_PREFILL_FORCE_LENGTH_CAP=1` 和 `VLLM_ROUTER_PD_REUSE_PROMPT_IDS=1`，重试关闭，最大并发1024。混合Router的5:9是两条路径的在途比，不是固定请求数量比；两层Router的完整参数分别在上述两个脚本。

## Mooncake Store的实际资源

共享L2总计3360GiB，SSD L3配额总计19936GiB，均由下面8个owner提供；L3配额不是当前已用空间。各owner的NUMA、CPU、网卡、目录、进程内存上限和offload参数均完整写在脚本中。

| owner脚本 | L2 / GiB | L3配额 / GiB | NUMA / CPUs | RDMA网卡 | SSD目录 |
| --- | ---: | ---: | --- | --- | --- |
| [92-0](store/mooncake-owner-92-0.sh) | 496 | 1328 | 0 / `32-39,160-167` | `mlx5_0,mlx5_6` | `/data/bench/eng-xinhao/kimi31-hisparse-store-r15/owner-0` |
| [92-1](store/mooncake-owner-92-1.sh) | 496 | 2272 | 0 / `40-47,168-175` | `mlx5_8,mlx5_10` | `/data2/bench/eng-xinhao/kimi31-hisparse-store-r15/owner-1` |
| [92-2](store/mooncake-owner-92-2.sh) | 560 | 2800 | 1 / `96-103,224-231` | `mlx5_12,mlx5_14` | `/data3/bench/eng-xinhao/kimi31-hisparse-store-r15/owner-2` |
| [92-3](store/mooncake-owner-92-3.sh) | 560 | 3184 | 1 / `104-111,232-239` | `mlx5_18,mlx5_20` | `/data4/bench/eng-xinhao/kimi31-hisparse-store-r15/owner-3` |
| [93-0](store/mooncake-owner-93-0.sh) | 320 | 1680 | 0 / `32-39,160-167` | `mlx5_0,mlx5_6` | `/data/bench/eng-xinhao/kimi31-hisparse-store-r15/owner-0` |
| [93-1](store/mooncake-owner-93-1.sh) | 320 | 2304 | 0 / `40-47,168-175` | `mlx5_8,mlx5_10` | `/data2/bench/eng-xinhao/kimi31-hisparse-store-r15/owner-1` |
| [93-2](store/mooncake-owner-93-2.sh) | 304 | 3184 | 1 / `96-103,224-231` | `mlx5_12,mlx5_14` | `/data3/bench/eng-xinhao/kimi31-hisparse-store-r15/owner-2` |
| [93-3](store/mooncake-owner-93-3.sh) | 304 | 3184 | 1 / `104-111,232-239` | `mlx5_18,mlx5_20` | `/data4/bench/eng-xinhao/kimi31-hisparse-store-r15/owner-3` |

owner `i` 的transfer/RPC/HTTP端口分别为 `62040+i`、`62140+i`、`62240+i`；owner0/1/2/3的transfer IP依次为 `11.14.0.<节点>`、`11.16.0.<节点>`、`11.1.0.<节点>`、`11.3.0.<节点>`。RDMA GID index=3，`MC_MAX_MR_SIZE=4294967296`。Master启用offload、evict时落盘、命中晋升，`promotion_admission_threshold=2`、`allocation_strategy=free_ratio_first`。

P使用 [store/mooncake-92.json](store/mooncake-92.json)：`standalone-store`、`P2PHANDSHAKE`、RDMA、Master `11.5.0.92:51561`，`global_segment_size=0`，`local_buffer_size=4GiB`。P本身不贡献共享Store L2，设置 `save_decode_cache=false`；两节点各4个owner服务同一个共享池。D启用HiSparseConnector和MooncakeConnector consumer，没有配置MooncakeStoreConnector。

## 从零恢复的顺序

脚本是当前92/93的路径快照；固定镜像、主模型、draft、网卡和宿主机挂载目录须已就绪。各脚本用 `--mount` 的源目录见脚本及清单。下面 `$ROOT` 指远端本目录：

```bash
ROOT=/data2/bench/eng-xinhao/vllm-pd/kimi3.1/b300-92-93-p8-d8-tp8-dcp8-ep8/hisparse-hybrid-5to9-tc66040
```

1. 在92启动Master，再在92/93分别启动本机的4个owner。

```bash
# b300-92
bash "$ROOT/mooncake-master-92.sh" --execute
for i in 0 1 2 3; do bash "$ROOT/store/mooncake-owner-92-$i.sh" --execute; done

# b300-93
for i in 0 1 2 3; do bash "$ROOT/store/mooncake-owner-93-$i.sh" --execute; done
```

1. 两节点准备引擎 `/work` 辅助文件。该目录实际挂载自下列 `WORK_ROOT`；[runtime-support.tgz](runtime-support.tgz) 原样收录两节点正在使用、SHA一致的 `pd_admin.py` 和 `clear_tokenizer_cache.py`，校验和见 [runtime-support.json](runtime-support.json)。前者提供命令里指定的 `pd_admin.CapacityObserver`；后者是本轮启动前的分词缓存清理工具。解包和安装适用于准备部署目录，未在此次文档补齐中执行。

```bash
# b300-92和b300-93，分别执行
WORK_ROOT=/data2/bench/eng-xinhao/hisparse-pd-correctness-20261113-ced31f9/full
mkdir -p "$WORK_ROOT/store-plan"
tar -xzf "$ROOT/runtime-support.tgz" -C "$WORK_ROOT"
install -m 644 "$ROOT/store/mooncake-92.json" "$WORK_ROOT/store-plan/mooncake-92.json"
```

当前仅P通过 `MOONCAKE_CONFIG_PATH=/work/store-plan/mooncake-92.json` 使用Store配置。模型与draft路径、缓存目录保持两侧脚本原值；P/D的draft路径不同，但文件SHA已验证一致。

1. 在92启动P、93启动D。

```bash
# b300-92
bash "$ROOT/prefill-92.sh" --execute

# b300-93
bash "$ROOT/decode-93.sh" --execute
```

1. 等两个模型健康后，在92依次启动原生PD Router、混合Router。

```bash
# b300-92，确认下面两个health均返回200后继续
curl -fsS http://11.5.0.92:19580/health
curl -fsS http://11.5.0.93:19581/health
bash "$ROOT/router-pd.sh" --execute
# 原生Router健康后，再启动混合入口
curl -fsS http://11.5.0.92:19500/health
bash "$ROOT/router.sh" --execute
curl -fsS http://11.5.0.92:19511/health
curl -fsS http://11.5.0.92:19511/routing_state
```

后续若替换P进程，P健康后需要刷新原生Router在启动时缓存的Mooncake engine ID和worker状态；命令为 `docker restart --timeout 20 kimi31-pd-hisparse-92-router-full-c256-r14`。刷新后确认 `http://11.5.0.92:19500/workers` 中P/D均为healthy，再做PD验证。

1. 服务链路就绪后启动压测。本轮已经由 `deploy_and_start.py` 完成清缓存验收、启动客户端及watcher；当前run_id和命令在 [STATE.json](STATE.json)。[benchmark.sh](benchmark.sh) 复用本轮trial，已有run目录时不能重复作为新一轮使用。`report.sh` 只读生成性能报告。

查看任一完整Docker命令，例如：

```bash
bash "$ROOT/router.sh" --dry-run
bash "$ROOT/router-pd.sh" --dry-run
bash "$ROOT/mooncake-master-92.sh" --dry-run
bash "$ROOT/store/mooncake-owner-93-0.sh" --dry-run
```
