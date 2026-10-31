# vLLM 0.32.0 / Router 0.1.16 · GLM-5.3 Flash

- **SM103 / SM120 NVFP4**：补齐 B300 与 RTX PRO 6000 的原生编译路径和算子注册。
- **FP8 KV 传输**：采用回滚后C230的原生PD与Store传输路径，不保留P-only实验的异构传输映射。
- **原生 MTP**：回滚 P-only、固定零 proposal、草稿槽释放和远端三槽映射实验，P/D 均保留原生 MTP3。
- **FlashInfer 调优**：修复 MoE 调优键包含 rank 导致的同步卡住，支持预调优后按 rank 加载配置。
- **Mooncake 缓存**：增加分组紧凑存储、异步查找、混合前缀复用及编码隔离命名空间；匹配粒度 256，淘汰水位 98%。
- **SM120 索引器**：回滚实验性融合 top-k 接入，Prefill / Decode 恢复原生路径。
- **阶段计时**：取消请求缺少调度或首输出事件时，不再产生负排队时间及异常 prefill 时间；缺失阶段单独计数。
- **Tokenizer 缓存镜像**：将原生 CPU 分词缓存打入原镜像，支持按 API 进程设置容量，修复单 API 指标注销，导出复用、新编码和缓存用量；修正原镜像 MLA 缓存写入的六参数接口不匹配。
- **共享分词与 API 亲和**：支持同机API共享精确token-ID池及私有L1，新增私有Unix socket和前缀/负载API代理；保留原生分词结果、DP rank及流式输出。
- **Store 远端读取与放置**：远端读取可独立按64MiB拆批；支持按前缀加权选择首选Owner并保留原生全局分配回退。部分读回失败使整个请求相关KV失效，重算后不反复读取同一坏对象。
- **DeepEP 通信预算**：支持通过AI_INFRA_DEEPEP_V2_NUM_SMS显式设置通信SM数量，未设置时沿用原生选择。
- **Store 读租约**：仅对租约过期的 GET 项重新查询副本并读取一次；保留原始失败和重试指标，不调整租约时长。
- **MLA / MTP**：合入 528 字节紧凑 KV 布局和能力检查下的 MTP 元数据复用。
- **Router**：修复 D 请求预占负载计数，覆盖流式结束与取消；补充历史前缀及 usage 处理，合并PD计算量/KV容量与耦合实例逐DP负载路由。
- **源码与构建**：在 integrations/glm53flash 保存固定版本补丁、依赖清单和镜像构建方法。
- **启动配置**：通用 1P2D、Router、Mooncake 命令见 DEPLOYMENT.md；P API4、KV接收线程4、异步查找开启，不传禁用自动调优的参数。

配套 Router 源码位于 tokmo-infra/router，同名分支：vllm-0.32.0-0.1.16-glm5.3-flash。

预调优的 C16/SWE13w 对照和复核结果见 [FLASHINFER-AUTOTUNE.md](FLASHINFER-AUTOTUNE.md)。
