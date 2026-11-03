# B300：8×DEP2 P + Wide EP16 D

生产配置固定 DSpark P5/D5；C7480/A12000。D 使用 verbs、MXFP8、direct staging、overlap、tile2048，GID3/QP4/TC106。P16+D16=32个GPU rank。Router 使用 cache_aware/round_robin、128GiB内存上限。

P/D/Store 明确使用 `xinhao-vllm-roce-megamoe:4c711ac-20261103`；Router 明确使用 `xinhao-vllm-router:e9b0421-tokenids-20261103`。完整ID和代码来源在 `images.env`。当前镜像是v0.32 backport：RoCE核心文件对应4c711ac，完整vLLM源码树与该分支不同。这里归档启动配置，不宣称已经通过D5整模型精度、稳定性或性能验收。

## 部署位置与前置检查

将本目录内容同步到四台节点的 `WORKDIR`（`cluster.env` 默认 `/data2/bench/eng-xinhao/vllm-pd/dsv41-flash-1017/b300-8p2-d16-rdma-megamoe-main`），在该目录执行命令。宿主机需有Docker/NVIDIA运行时、Python3、RDMA设备、模型目录及配置中的SSD挂载。镜像必须已存在，不自动拉取。示例私网IP和HCA来自当前89/91/92/93实机拓扑，换节点须重新核验。

- P/D与Store显式设置 `MC_ENABLE_DEST_DEVICE_AFFINITY=1`。用户在2026-11-03确认，缺少此参数会导致本轮HTTP500；启动后用Docker inspect核验实际Env。
- D两节点须重新读回GPU/NIC路径ACS与GPU DMA-BUF RDMA验收结果；脚本不会修改PCIe、PFC或QoS。
- 先加载模型，再分配CPU Store。逐NUMA核验物理余量和零owner换页，预留P分词缓存、Router、客户端及64GiB系统余量。
- CPU L2共7104GiB，SSD L3配额42176GiB；配置见 `nodes/*.env`。保留Store独立命名空间。首次部署先创建脚本中列出的SSD owner目录。
- Router先检查全部P/D入口，再核验发现16P+16D健康worker。

## 启动顺序

```bash
# 89：master
bash mooncake-master.sh

# 89和91各执行，NODE分别填89、91
for i in 0 1 2 3; do bash prefill.sh NODE "$i"; done

# 92和93分别执行
bash decode.sh 92
bash decode.sh 93

# 模型完全加载且NUMA预算通过后，四台各执行，NODE填本节点号
for i in 0 1 2 3; do bash mooncake-owner.sh NODE "$i"; done

# 全部后端健康后，在89执行
bash router.sh
bash benchmark.sh
```

命令示例中的NODE是待替换值。D92提供全局16-rank API入口，D93为headless节点。`decode.sh` 的profiler选项默认关闭；日常部署不设置 `D_PROFILE_DIR` 或 `D_PROFILE_SOURCE`。

## 压测

`benchmark.yaml` 通过原生客户端固定C7480/A12000、600秒请求deadline、零重试；HTTP错误记录后继续，不安装首错停服watchdog。输入包含缓存token，输出使用最终usage，正式吞吐从首遍轮次40%且并发达标后取完整分钟。

`benchmark.sh` 复用封存的12000会话/1196074参考轮CPU语料，核对manifest和数据SHA，不重新准备约556GiB正文。`CLIENT_HOME` 下既有客户端源码、observer依赖、封存缓存及数据集须存在；本包不复制语料、模型、镜像、日志或凭据。

一次运行使用独立容器名和结果路径，脚本拒绝覆盖已有DB。下一次压测须同时更新 `benchmark.yaml` 的saver路径，以及 `benchmark.sh` 中对应容器名、BENCH_RUN_DIR和诊断路径。启动前另行完成GPU/外部KV/CPU分词缓存/Router前缀树的冷缓存核验，并保存同窗服务端和实际in-flight轨迹；本脚本本身不执行缓存清理。
