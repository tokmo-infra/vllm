# 基于 B300 Kimi K3.1 现用镜像进行二次开发

本分支以 B300 当前使用的 `vllm/vllm-openai:kimi-k31` 为基础；现用镜像的
本地 ID 为 `sha256:1a8a702f2bfeded3c90259ae60fefbd9e4ca014241946e7d5f3396badce99e76`，
对应的 vLLM 源码基线为 `266e88c051ca46607fd9de20216bd58426f80052`。

## Python / Triton 源码开发

在有上述基础镜像的 B300 机器上，从仓库根目录执行：

```bash
docker image inspect sha256:1a8a702f2bfeded3c90259ae60fefbd9e4ca014241946e7d5f3396badce99e76
docker build --network=none --pull=false -f docker/Dockerfile.kimi-k31 \
  -t local/vllm-kimi-k31:dev .
```

这个入口保留基础镜像的依赖、原生扩展、入口和运行配置，把 `vllm/` 中相对
基线新增、修改或删除的源码应用到新镜像。包括 Python 写的 Triton/CuTe 代码；
镜像里不属于仓库基线的生成文件和第三方文件不会因为同步而整目录删除。
基础镜像和原有标签不会被改变。使用独立的新标签，不要覆盖正在使用的标签。

镜像包含构建记录，可以查看源码摘要及增、改、删的文件列表：

```bash
docker run --rm --network=none --entrypoint cat local/vllm-kimi-k31:dev \
  /opt/kimi-k31-build/source-update.json
```

构建使用当前 Docker 上下文，而不是只使用 Git 提交；**未提交和未被
`.dockerignore` 排除的文件也会进入镜像**。建议构建前检查 `git status`，
并核对构建记录。镜像内 `vllm.__version__` 仍为底座版本，不能用它区分开发
版本；请通过镜像标签、镜像 ID 和上述构建记录区分。

## 原生扩展和依赖的限制

此入口**不编译 C/C++、CUDA 或 Rust 扩展，也不安装新的 Python 依赖**。
现用镜像虽然自带 CUDA 编译器，但缺少完整构建工具和外部源码，尚未验证一套
可直接重编所有原生扩展的流程。修改 `csrc/`、`cmake/`、`rust/`、构建工具、
依赖清单或 `vllm/` 中的原生源码后，当前入口会报错并列出变化，避免把旧
二进制误当成已经包含新代码。需要另行准备原生构建环境、编译和验证后再
发布；这个入口目前不能完成这类改动。

构建通过仅说明源码同步和上述检查通过，并不代表模型推理、性能或数值正确性
已经验证。上线前应针对改动执行对应的单测、算子测试和 B300 模型推理验证。
