# Tokmo 企业自定义推理镜像

企业文档、部署配置、构建资料和专项验证统一存放在本目录，按模型、硬件和部署拓扑组织。vLLM 包内的实际源码仍位于根目录 `vllm/`，便于构建、使用和比较上游改动。

| 内容 | 入口 |
| --- | --- |
| DeepSeek-V4.1-Flash / B300 / 4×DEP2 P/D 耦合 | [部署文档](docs/deepseek_v41_flash/b300/pd-coupled-4xdep2.md) |
| 4×DEP2 耦合配置与启动脚本 | [部署目录](deployments/deepseek_v41_flash/b300/pd-coupled-4xdep2/README.md) |
| 镜像构建与来源校验 | [Dockerfile](images/deepseek_v41_flash_b300/Dockerfile)、[来源信息](images/deepseek_v41_flash_b300/image-info.json) |
| 专项验证 | [验证说明](tests/README.md) |

当前仅维护 8×B300、4×TP1/DP2/EP2、1600GiB L2 和 2TiB L3 的生产推荐配置。

## 目录命名

部署配置采用 `deployments/<模型>/<硬件>/<部署拓扑>/`，文档采用 `docs/<模型>/<硬件>/<部署拓扑>.md`。镜像构建资料放在 `images/`，供适用的部署拓扑复用。

当前拓扑为 `pd-coupled-4xdep2`，表示 4 个 DEP2 的 P/D 耦合实例，每实例 TP1/DP2/EP2。后续 16×DEP2 + wide EP32 的 P/D 分离配置可并列命名为 `pd-disagg-16xdep2-wideep32`；目录创建时再填写对应的实际配置。

```text
tokmo/
├── deployments/deepseek_v41_flash/b300/
│   └── pd-coupled-4xdep2/
├── docs/deepseek_v41_flash/b300/
│   └── pd-coupled-4xdep2.md
└── images/deepseek_v41_flash_b300/
```
