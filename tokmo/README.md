# Tokmo 企业自定义推理镜像

企业文档、部署配置、构建资料和专项验证统一存放在本目录，按模型和硬件组织。vLLM 包内的实际源码仍位于根目录 `vllm/`，便于构建、使用和比较上游改动。

| 内容 | 入口 |
| --- | --- |
| DeepSeek-V4.1-Flash B300 生产部署说明 | [部署文档](docs/deepseek_v41_flash_b300.md) |
| 生产配置与启动脚本 | [部署目录](deployments/deepseek_v41_flash/b300/README.md) |
| 镜像构建与来源校验 | [Dockerfile](images/deepseek_v41_flash_b300/Dockerfile)、[来源信息](images/deepseek_v41_flash_b300/image-info.json) |
| 专项验证 | [验证说明](tests/README.md) |

当前仅维护 8×B300、4×TP1/DP2/EP2、1600GiB L2 和 2TiB L3 的生产推荐配置。
