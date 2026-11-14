# B300 HiSparse PR66040候选：P/D统一镜像

完整服务清单与逐节点启动命令见 [DEPLOYMENT.md](DEPLOYMENT.md)：两层Router、Mooncake Master、8个Store owner、P使用的Store配置及/work辅助文件。13个容器与实跑命令逐项核对，519项检查通过；默认dry-run。

P、D启动脚本固定使用同一image digest，角色差异由参数、环境与挂载配置。镜像以原D底包为基线，只加入c80c339d6的三个KDA文件；相对原P还包含仅对DCP HiSparse生效的容量保护。原容器保留用于回退。

P原来是通用decode，候选同时启用latency_specialized并关闭不兼容的GEMM_RS；保留P的prefix/align、Store及prefill批次。因此整体对比不等于单独66040收益。D保持原专用decode配置。

CPU59项、B300基线51项、候选扩展93项通过，阈值未放宽。单层R185耗时下降28.3%、R192下降32.1%，不能换算为整机TPM。全模型与压测状态见deployment.json。C288/A576、5:9、输出2048与原基线一致。

Dockerfile的构建上下文只放对应提交的prepared.py、fused_tc.py、tc_policy.py，保留vllm目录层级；禁止把整个源码树复制到原镜像。先在构建节点将上述base_image标记为local/kimi31-hisparse-d-r17:baseline-20261114。部署ID及导出归档校验和见deployment.json。

混合Router实现仍属于router/xinhao-vllm-router，未复制实现或改动Router。

P进程替换后需重启原生CPU Router以刷新其启动时缓存的Mooncake engine ID和worker健康状态；先确认P/D就绪，再确认Router的两个worker均healthy，最后做PD回归。无需重启Store或混合Router。
