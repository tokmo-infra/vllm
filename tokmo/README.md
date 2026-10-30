# Tokmo 企业部署形态

本目录的下一层按部署形态分类。每种形态独立收纳相关文档、启动脚本、镜像构建资料和专项验证。

| 部署形态 | 入口 | 当前配置 |
| --- | --- | --- |
| 4×DEP2 P/D 耦合 | [pd-coupled-4xdep2](pd-coupled-4xdep2/README.md) | DeepSeek-V4.1-Flash、8×B300、1600GiB L2、2TiB L3 |

```text
tokmo/
├── README.md
└── pd-coupled-4xdep2/
    ├── README.md
    ├── docs/
    ├── deployments/
    ├── images/
    └── tests/
```

后续 16×DEP2 + wide EP32 的 P/D 分离配置可使用同级入口 `pd-disagg-16xdep2-wideep32/`，并将它的资料集中存放在该目录中。
