# B300 Kimi K3.1 现用镜像基线

本分支以当前 B300 上的 vllm/vllm-openai:kimi-k31 为准，其镜像 ID 为
sha256:1a8a702f2bfeded3c90259ae60fefbd9e4ca014241946e7d5f3396badce99e76，
对应的 vLLM 源码提交为 266e88c051ca46607fd9de20216bd58426f80052。
分支保留最新 main 的历史，但源码树采用现用镜像的基线，不包含随后的主线改动。

在本机已有上述镜像 ID 的 B300 上，从仓库根目录执行：

    docker image inspect sha256:1a8a702f2bfeded3c90259ae60fefbd9e4ca014241946e7d5f3396badce99e76
    docker build --network=none --pull=false -f docker/Dockerfile.kimi-k31 -t local/vllm-kimi-k31:baseline .

构建先对源码清单做 SHA-256 校验，再将镜像内存在的对应源码与仓库逐文件比对。
校验不通过时构建失败；镜像中由 wheel 打包生成或额外携带的第三方文件及原生
二进制继续使用原镜像内的文件。最终镜像继承原镜像的层、入口及环境，仅额外
添加 /opt/kimi-k31-baseline/verified.json 校验记录，镜像 ID 因此不同。

**这不是从源码独立重新编译镜像。** 构建依赖上述本地基础镜像；如果要让后续
源码改动进入镜像，须先恢复并固定原构建上下文、依赖和原生扩展的构建流程，
重新构建并验证，不能依赖这个基线打包入口。不要修改清单来绕过校验。
