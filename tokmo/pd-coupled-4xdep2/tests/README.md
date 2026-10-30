# 企业镜像专项验证

`test_dsv41_native_cache.py` 使用本地真实 DSV4.1 tokenizer，验证原生 renderer 的精确 token ID、added-token 边界、Unicode、namespace 隔离、LRU 容量、epoch 失效、同步/异步传递及原生回退行为。无模型权重计算，无生成或 GPU 显存分配。

在 B300 仓库根目录运行（镜像内已具备 pytest）：

```bash
docker run --rm --network none \
    --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=none \
    -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -v /data2/models/DeepSeek-V4.1-Flash-1017:/model:ro \
    -v "$PWD/tokmo/pd-coupled-4xdep2/tests:/verification:ro" \
    --entrypoint /bin/bash \
    local/vllm-openai:0.32.0-dsv41-native-token-cache-v1 \
    -lc 'cd /tmp && uv venv --python /usr/bin/python3.12 --system-site-packages /tmp/.venv && /tmp/.venv/bin/python -m pytest -q -c /dev/null -p no:cacheprovider /verification/test_dsv41_native_cache.py'
```

三份镜像源码的 SHA256 固定在 `tokmo/pd-coupled-4xdep2/images/deepseek_v41_flash_b300/source.sha256`。重建 Dockerfile 会核对它们并执行 Python 编译。为保留原镜像的精确源码，这三份文件不做格式化重写；新增部署工具和验证代码按仓库风格检查。

这些验证不替代完整容量 L2/L3 配置的模型验收和性能测量。

本次[验证记录](validation.json)：镜像重建后 5440 个 vLLM 安装文件的哈希全部与原镜像一致，9 项 tokenizer/renderer 回归通过，14 个生产容器的启动参数完成无服务启动的静态核对。
