# Changelog

所有版本作者：晨星 (CJX0712)。

## [0.1.0] - 2026-10-11

### Added
- 确定性可复现系统：`core/seed.set_all` + 独立于全局状态的种子派生（修复 `hash()` 跨进程不确定性）。
- 合成数据 DGP：`gaussian` 高斯混合 + `moons` Two Moons 非线性流形 + 高维冗余噪声特征。
- 防泄漏切分：测试集全标注、训练集无标注。
- 半监督模型：LabelSpreading(kNN 图)、SelfTraining(XGB)、纯 numpy 标签传播（离线兜底）。
- 纯监督基线：LogisticRegression、XGBoost、RandomForest。
- 旗舰 `SemiFuse`：标注预算自适应路由 + 诚实回退。
- 端到端基准 `cli.py demo`：确定性校验 + 消融 + 诚实结论，落盘 `benchmark.json`。
- 46 个 pytest 用例（核心模块 90% 行覆盖）、ruff 检查、GitHub Actions CI。
- 交付骨架：requirements / Dockerfile / Makefile / pyproject / README / docs / references。

### Fixed
- 确定性：`hash(dgp)` → 稳定 `from_bytes` 编码（跨进程可复现）。
- LabelSpreading 改用 kNN 图（RBF 全连接在流形交错区失效）。
- 图方法/LP 移除 StandardScaler（冗余维被放大破坏距离）。
- median heuristic 自适应 gamma（固定 gamma=10 致亲和矩阵退化）。
- 行归一化转移矩阵 W 替换对称归一化 P + α=0.99（避免发散）。
- sklearn 1.9 `base_estimator` → `estimator`。
- SemiFuse 主 fit 补候选 fit（修复 NotFittedError）。
- 废弃少样本 OOF 路由（失真），改标注预算自适应。
- derive_failures 聚合口径修正（限定方法过滤）。

### Methodology
- 诚实结论：SemiFuse 胜强非线性监督（XGB/RF），与正则化线性监督持平；充足标注诚实回退零负增益。
