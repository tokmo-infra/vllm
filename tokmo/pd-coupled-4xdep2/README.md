# 4×DEP2 P/D 耦合部署

本目录集中存放这套 P/D 耦合部署的说明、启动脚本、镜像构建资料和专项验证。当前配置为 DeepSeek-V4.1-Flash-1017、单机 8×B300、4×TP1/DP2/EP2、1600GiB 共享 L2 和 2TiB SSD L3。

| 内容 | 入口 |
| --- | --- |
| 模型与硬件部署说明 | [DeepSeek-V4.1-Flash / B300](docs/deepseek_v41_flash_b300.md) |
| 生产配置与启动脚本 | [部署目录](deployments/deepseek_v41_flash/b300/README.md) |
| 镜像构建与来源校验 | [Dockerfile](images/deepseek_v41_flash_b300/Dockerfile)、[来源信息](images/deepseek_v41_flash_b300/image-info.json) |
| 专项验证 | [验证说明](tests/README.md) |

从仓库根目录构建镜像：

```bash
bash tokmo/pd-coupled-4xdep2/deployments/deepseek_v41_flash/b300/build-image.sh
```

部署脚本与文档中的命令以本目录的实际路径为准；vLLM 包内源码保持在仓库根目录的 `vllm/`。
