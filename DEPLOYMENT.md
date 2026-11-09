# GLM5.3 Flash 1103 · 部署入口

- [1P2D部署](deployment1p2d.md)：1台双EP4 P、2台EP8 D，合计24GPU；D保持seq16 / graph64。
- [4P4D部署](deployment4p4d.md)：4台双EP4 P、4台EP8 D，合计64GPU；D为seq32 / graph128。

两份部署文档开头说明NVFP4融合、P侧MTP精简、Router交接和缓存字段的作用与使用示例。P/D均保留原生异步、CUDA图、默认注意力后端与Mooncake缓存；Router使用P cache_aware / D round_robin。
