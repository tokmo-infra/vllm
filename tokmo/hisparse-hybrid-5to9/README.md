# 待实测：C288 / A576，P本地:PD = 5:9

复用原P/D、原生PD Router及八个Store owner；新建CPU入口92:19511。
满载目标约103本地/185PD。输出上限2048；不按输入/输出长度分流。

## 源码与部署

- 引擎PR66040已通过`75d25d689`回退，不重建或重启已验证模型镜像。
- P源码`bab602a8`，D源码`5caa5d7f`；逐文件运行时哈希见上级`runtime-code-audit.json`。
- 原生Rust PD Router源码`96885611`，现有二进制不变。
- 混合Router实现与启动脚本：[router提交](https://github.com/tokmo-infra/router/tree/4e4db92997074544577a33b97b579e535fd2a04b/tokmo/kimi31-hisparse-hybrid)。
- 从该目录执行`bash router.sh --execute`，只创建CPU入口。
- `benchmark.yaml`为完整本轮配置，原生perf-benchmark使用`run benchmark.yaml`启动。
- 首次启动前依次清GPU KV、共享Store L2/L3、P/D CPU分词缓存，并核实回执。

本地完整协调脚本和报告工具位于ai-infra笔记同名目录；机器部署在
`/data2/bench/eng-xinhao/vllm-pd/kimi3.1/b300-92-93-p8-d8-tp8-dcp8-ep8/hisparse-hybrid-5to9`。

## 统计与保护

40%且并发达标后，首个平均输入150k..160k完整分钟起取连续10完整分钟，
整窗平均仍需在该范围。按长度选择，不挑TPM峰值；随后原生SIGINT取消尾部并归档。
P活跃KV达到97%、D达到94%持续30秒（启动120秒后），或推理失败、服务退出时停止客户端。
不清运行中的缓存，不修改模型参数。新配置尚无成绩，不把容量估算当吞吐提升。

HBM使用`vllm:kv_cache_usage_perc`活跃KV池口径，P/D每卡池容量46.328/44.896GiB；
排除free队列可淘汰前缀，不使用整卡显存占用代替。报告包含当前值、同窗时间均值和峰值。
