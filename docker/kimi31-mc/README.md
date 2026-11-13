# Kimi K3.1 MC 开发镜像

本目录将列出的 Python 源文件覆盖到固定 MC 镜像上，保留原镜像的 CUDA
内核与原生 Mooncake SDK，**不声称原生二进制已由本仓库重新构建**。
SDK 校验要求 CUDA 13 的 mooncake-transfer-engine-cuda13==0.3.14rc3。
Dockerfile 只覆盖明确列出的文件；其它 Python 改动须补入清单，原生改动须完整构建。

在仓库根目录构建：

    docker build -f docker/kimi31-mc/Dockerfile.runtime --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" -t vllm/vllm-openai:kimi-k31-mc-kvfix-yyf .

离线机器可用 BASE_IMAGE 指向已核验的固定 MC 镜像或仅加分词缓存的派生版本。
记录真实 image ID、基础镜像 digest 与源码载荷摘要；不能直接用变化中的
kimi-k31-mc 标签替代固定 digest。

镜像设置统一的 16 GiB CPU 分词缓存池、每 API 进程 64 MiB L1。
这与 Mooncake 的模型 KV 存储是两个独立缓存。
MC 镜像默认设置 VLLM_MOONCAKE_STORE_SELECTIVE_LEASE=1，
需要旧租约路径时可显式设置为 0。PUT 压力恢复及 GET 过期租约恢复默认生效，
只有对应失败才触发，正常读写不增加重试。
兼容的 rank-local DCP MLA 默认启用 compact_partial_blocks，
可在 kv_connector_extra_config 中设置为 false 关闭；普通 TP、非 MLA、
异构 Store TP 或 HiSparse 不因省略该参数而启用裁剪。适用布局与配置见
[Mooncake Store 使用说明](../../docs/features/mooncake_store_connector_usage.md)。

小批长上下文验证与限制见[验证记录](VALIDATION.zh-CN.md)。
必须区分实际少存与仅少读；小批回复和字节一致性不等于通用模型精度评测，
TTFT 相近也不能掩盖后台 PUT 的回退。
