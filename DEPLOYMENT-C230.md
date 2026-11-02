# C230 历史配置

原C230使用一个P角色（两个TP1/DP4/EP4）及两个D角色（各TP1/DP8/EP8），P/D显式CX8、FP8 KV、MTP3、prefix256，三个750GiB Store、水位98%。

SWE8w、C230/U345/N345、55%后12个完整分钟，原报告合计5112.04万TPM。

当前通用启动命令统一见根目录 [DEPLOYMENT.md](DEPLOYMENT.md)，包括P API4、KV接收线程4、异步lookup和默认自动调优。

后续共享 Tokenizer、撤 API 代理复核 run-8bb8d266fb55 为 SWE8w C230/U345/N345、55% 后完整窗口 5595.45 万 TPM；该历史结果不是新 SWE13w C200/U300/N300 的测量结果。当前 P81/D83/D84 配置及逐机共享镜像 ID 以 [DEPLOYMENT.md](DEPLOYMENT.md) 为准。
