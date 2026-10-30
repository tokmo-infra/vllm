# DeepSeek-V4.1-Flash on B300

本目录提供 8×B300、4×TP1/DP2/EP2、共享 1600GiB L2 和 2TiB SSD L3 的生产推荐配置。

镜像来源、构建、启动顺序、KV 卸载、HCA/NUMA、端口和维护说明统一见 [部署文档](../../../docs/deepseek_v41_flash_b300.md)。

启动脚本默认使用自身目录保存状态，可通过 `WORKDIR` 指定状态目录。该目录不包含历史实验或压测配置。
