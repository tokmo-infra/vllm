# GLM-5.3 Flash 源码与构建

- integrations/glm53flash/manifest.json 固定vLLM 0.32.0、FlashInfer 0.7.1、Router 0.1.16及补丁哈希。
- prepare_sources.py 还原固定源码并应用Python和native补丁；不再提供P-only实验选项。
- P/D采用C230回滚后的完整原生MTP状态分配，移除固定零proposal、草稿槽释放和远端三槽映射实验。
- FlashInfer回移#6006/#6139，删除调优缓存键中随rank变化的字段，保留并行大小、专家数量和形状。
- 预调优加载通过GLM_FLASHINFER_PRETUNED_ONLY选择；普通启动保持原生自动调优，不依赖/experiment目录。
- 实验性SM120融合索引器及入口已删除，恢复原生Prefill/Decode路径。
- 保留Mooncake分组紧凑存储、异步查找、GET租约过期后仅重试一次、有效阶段计时。
- Router合并PD计算量/KV容量路由及耦合实例逐DP KV/负载路由。

## 还原源码

~~~bash
.venv/bin/python integrations/glm53flash/prepare_sources.py sources
~~~

原生构建配方见integrations/glm53flash/Dockerfile。整合镜像保留CUDA、Torch和已编译扩展，以回滚后的运行时代码和调优修复更新Python层；构建记录保存镜像ID、源码与二进制哈希。

通用1P2D、Router、Mooncake配置见 [DEPLOYMENT.md](DEPLOYMENT.md)。C16调优验证见 [FLASHINFER-AUTOTUNE.md](FLASHINFER-AUTOTUNE.md)。

## 整合镜像

- 推理：local/glm53flash-vllm:0.32.0，镜像ID 540155b18ee7。
- Router：local/glm53flash-router:0.1.16，镜像ID 21c9cfa19e34。
