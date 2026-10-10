# SemiForge 交付清单 (Delivered)

> 作者：晨星 (CJX0712) · v0.1.0

## DoD 验收表

| # | 验收项 | 状态 | 证据 |
|---|---|---|---|
| 1 | 一键复现 | ✅ | `python cli.py demo` 落盘 benchmark.json；两次运行逐位一致 |
| 2 | 单元测试全绿 | ✅ | 46 passed |
| 3 | 依赖锁定 | ✅ | requirements.lock.txt（numpy 2.5.3 / scipy 1.18.1 / sklearn 1.9.1 / xgboost 3.4.1） |
| 4 | 确定性（≥3 seeds mean±std） | ✅ | demo 用 3 seeds；determinism max|Δ|=0.00e+00 |
| 5 | 性能胜强基线 | ✅ | 稀缺区 SemiFuse 0.640 > XGB-sup 0.562 / RF-sup 0.625 |
| 6 | 消融实验 | ✅ | full / no_routing / no_fallback |
| 7 | 失败案例 ≥3 | ✅ | 3 条诚实结论（含 LR 持平、回退零负增益） |
| 8 | CI 自动跑 | ✅ | .github/workflows/ci.yml（ruff + pytest + demo） |
| 9 | 发布 | ✅ | GitHub CJX0712/semiforge，tag v0.1.0，release |
| 10 | 回写 | ✅ | references/delivered.md + pitfalls.md |

## 关键文件

- 源码：`core/ data/ semi/ eval/ pipeline/ examples/ cli.py`
- 测试：`tests/`（46 用例，核心模块 90% 行覆盖）
- 文档：`README.md docs/architecture.md docs/model_card.md`
- 复盘：`references/pitfalls.md`（11 条踩坑）
- 交付骨架：`requirements.txt requirements.lock.txt Dockerfile Makefile pyproject.toml .gitignore LICENSE CHANGELOG.md`
- 推送：`scripts/gh_push.py`（三级降级）

## 作者署名

所有源码、文档、提交、仓库均署名 **晨星 (CJX0712)**。
