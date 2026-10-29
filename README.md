# vLLM 0.32.0 · GLM-5.3 Flash

- **SM103 / SM120 NVFP4**：补齐 B300 与 RTX PRO 6000 的原生编译路径和算子注册。
- **FP8 KV 传输**：兼容 P 端标准 FP8 与 D 端 packed MLA 布局，恢复跨架构 PD 的 MTP 接受率。
- **GPU KV 准入**：增加 P-only 选项，减少 Prefill 阶段不使用的目标验证草稿槽，提高可接入请求数。
- **KDA 状态传输**：识别组合 KV 分组，仅传输有效状态，减少草稿槽的冗余搬运。
- **Mooncake 缓存**：增加分组紧凑存储、异步查找、混合前缀复用及编码隔离命名空间；匹配粒度 256，淘汰水位 98%。
- **SM120 索引器**：合入 Prefill / Decode 索引优化及 GLM 适配。
- **MLA / MTP**：合入 528 字节紧凑 KV 布局和能力检查下的 MTP 元数据复用。
- **Router**：修复 D 请求预占负载计数，覆盖流式结束与取消；补充历史前缀及 usage 处理。
- **源码与构建**：在 integrations/glm53flash 保存固定版本补丁、依赖清单和镜像构建方法。
- **启动配置**：C230 三机 P、D、Router、Mooncake 的 CX8 命令统一整理在 integrations/glm53flash/DEPLOYMENT.md。
