# Kimi K3.1 B300 源码构建

本分支的 vLLM 源码基线为 `266e88c051ca46607fd9de20216bd58426f80052`，
B300 现用运行镜像为 `vllm/vllm-openai:kimi-k31`。92 和 93 上该镜像的
本地镜像 ID 均为：

```text
sha256:1a8a702f2bfeded3c90259ae60fefbd9e4ca014241946e7d5f3396badce99e76
```

## 构建

使用仓库原有的 `docker/Dockerfile` 和 `kimi-k31` 目标。它沿用 vLLM 的
`csrc-build`、`rust-build` 和 `build-wheel` 阶段，从当前 Docker 构建上下文编译
C++、CUDA、Rust 并组装 vLLM wheel，再将该 wheel 安装到现用 Kimi 运行镜像。
最终镜像继承现用镜像的其他依赖、环境和入口，不会覆盖原镜像标签。

构建机需要 Docker BuildKit/buildx、当前 Kimi 基础镜像，以及访问构建阶段
所需基础镜像和依赖源的网络或可用镜像。克隆本分支后，在仓库根目录执行：

```bash
BASE=vllm/vllm-openai:kimi-k31
EXPECTED=sha256:1a8a702f2bfeded3c90259ae60fefbd9e4ca014241946e7d5f3396badce99e76
docker buildx inspect default | grep -qE '^Driver:[[:space:]]+docker$' && \
test "$(docker image inspect --format '{{.Id}}' "$BASE")" = "$EXPECTED" && \
docker buildx build --builder default --load -f docker/Dockerfile --target kimi-k31 \
  --build-arg KIMI_RUNTIME_IMAGE="$BASE" \
  --build-arg max_jobs=16 --build-arg nvcc_threads=1 \
  -t local/vllm-kimi-k31:dev .
```

`max_jobs` 和 `nvcc_threads` 应根据机器空闲资源调整；默认采用原 Dockerfile
的 CUDA 架构列表，完整编译可能耗时较长、占用较多空间。不要传入
`VLLM_USE_PRECOMPILED=1`：`kimi-k31` 目标要求从源码构建。原 Dockerfile
后半段为组装 wheel 使用的中间产物，是**同一次构建**的 C++/CUDA 和 Rust
产物，并非从现用镜像或外部下载的旧 vLLM 扩展。
示例使用与 `docker image inspect` 相同 Docker daemon 的 `default` builder；
使用其他 builder 前应确认它能访问校验过的基础镜像。
此目标不执行发布到 PyPI 时的 wheel 大小上限检查，也不重新构建或替换
现用镜像中的额外 EP 扩展。

构建会包含未提交、未被 `.dockerignore` 排除的文件。建议先检查
`git status`。对于 Git 忽略的第三方生成目录，构建会拒绝未跟踪的残留文件，
以免上一次构建产生的代码覆盖本次依赖；若确实需要修改并提交这些源码，
应先用 `git add -f` 将其显式纳入版本管理。镜像的提交标识默认是
`unknown`，避免误用底座的旧提交号；
确认工作区干净后可用 `--build-arg VLLM_BUILD_COMMIT="$(git rev-parse HEAD)"`
记录当前提交。需要在其他构建机使用时，先提供相同的基础镜像或修改
`KIMI_RUNTIME_IMAGE` 指向可访问的同一镜像。这个本地 image ID 不能直接从
GitHub 仓库下载，也不能作为 BuildKit 的 `FROM` 镜像名；如使用镜像仓库，
可传入 `仓库/镜像@sha256:清单摘要` 固定仓库中的镜像。

## 构建检查与运行

安装新 wheel 前会核对构建环境和现用镜像的 Python、Torch、CUDA 和 Torch
C++ ABI；还会检查新 wheel 包含现用镜像的原生扩展和 Rust 前端，并检查新增
或变化的依赖（包括 extras 和传递依赖）是否已由现用镜像满足。wheel 组装时
会按摘要核对构建上下文中的源码、配置文件和本次 C++/CUDA wheel 中的全部原生
库（文档和构建生成的版本文件除外），避免源码被覆盖或新增扩展遗漏。
新增运行时资源若不在 wheel 中会报错，需要把它加入 wheel 的 package data。
为避免沿用旧扩展，会先卸载旧 vLLM
并清除其包目录，再安装新 wheel，不自动升级其他运行依赖。若修改引入新的
依赖而底座不满足（包括新增的可选 extra 依赖），需先更新运行底座或自行
处理依赖，不应忽略构建错误。
新增 C++/CUDA 或 Rust 文件仍需像常规 vLLM 开发一样加入对应构建目标，
单纯放入未引用的文件不会自动编译。

构建产物中可查看 wheel 摘要、各原生文件摘要、构建/运行 ABI 以及新旧
`pip check` 结果：

```bash
docker run --rm --network=none --entrypoint cat local/vllm-kimi-k31:dev \
  /opt/kimi-k31-build/source-build.json
```

现用镜像本身已有依赖元数据冲突，因此应比较记录中的新旧检查结果。构建
通过不等于模型功能、数值和性能已验证：部署前应使用实际 B300 启动参数
测试修改涉及的算子、Kimi 模型推理以及所需的性能表现。
