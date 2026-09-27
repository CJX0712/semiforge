# SemiForge · 模块化半监督学习系统

> 作者：晨星 · 2026-09-28 · CPU-only / 确定性可复现 / 零下载离线兜底

[![CI](https://github.com/CJX0712/semiforge/actions/workflows/ci.yml/badge.svg)](../../actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

SemiForge 是一套**模块化半监督学习（SSL）系统**：在极低标注预算（5%~10% 标签）下，
用图传播先验 + 特征模型集成做校准融合，配合类平衡课程化伪标签与守护回滚机制，
把无标注样本的流形/聚类结构转化为分类增益。

## 核心结论（先看这里）

在 6 个数据集 × 5 策略、统一评测口径（transductive，只评无标注点）下：

| 数据集 | 监督基线 | 旗舰 CBCP-Hybrid | 增益 | 判定 |
|--------|---------:|----------------:|-----:|------|
| moons      | 0.7684 | 0.8491 | **+0.0807** | ✅ 显著 |
| circle     | 0.5351 | 0.8228 | **+0.2877** | ✅ 显著 |
| spiral     | 0.4070 | 0.5158 | **+0.1088** | ✅ 显著 |
| digits     | 0.8578 | 0.9356 | **+0.0778** | ✅ 显著 |
| blobs      | 0.9193 | 0.9053 | −0.0140 | ⚠️ 噪声带内非劣 |
| highdim    | 0.9895 | 0.9860 | −0.0035 | ⚠️ 噪声带内非劣 |
| **均值**   |        |        | **+0.0896** | ✅ |

- 4/6 数据集显著为正；2/6 在 ±2·SE（≈±0.024）噪声带内非劣
- Optuna TPE 调参后（moons/circle/spiral）：circle 增益提升至 **+0.3491**
- 全部数字固定 seed=42，逐位可复现；详见 `examples/benchmark.json`

## 旗舰算法：CBCP-Hybrid（Calibrated fusion + Balanced Curriculum Pseudo-labeling）

三个机制叠加：

1. **标记集交叉校准的双源融合** — 图传播后验（kNN 图 + Zhu et al. 2002 迭代钳位）
   与特征模型集成（RF + LR）后验，各自在 labeled 集上做 k 折 CV 计分；
   融合权重 = 校准精度的幂次锐化 softmax（power=14）。
   近平局（|Δacc|≤0.06）时直接锚定特征源，**按构造非劣**于监督基线。
2. **类平衡课程伪标签** — 每轮配额 = base_quota × r，类间独立截断，
   置信下限 = max(类内分位数阈值, 0.5)，多数类永远不会淹没课程。
3. **守护回滚** — 每轮提交后用同一 CV 口径复评融合精度，**严格改进才保留**，
   否则整轮回滚并终止（防止伪标签自证偏置污染标签集）。

## 五策略对照

| 策略 | 一句话 | 后端 |
|------|--------|------|
| `supervised` | 仅标注集 LR，非劣锚点 | sklearn / 纯 numpy 兜底 |
| `self_training` | 类平衡迭代伪标签 | RF（sklearn）/ numpy LR |
| `graph` | kNN 图标签传播 | sklearn LabelSpreading / **纯 numpy Zhu 2002** |
| `co_training` | 双视图 Blum-Mitchell 协同训练 | RF / numpy LR |
| `cbcp_hybrid` | 校准融合 + 课程伪标签 + 守护回滚（旗舰） | 全部 |

**离线兜底**：sklearn 不可用时自动降级为纯 numpy 实现
（`NumpyLogisticRegression` softmax+Adam、kNN 图传播），零下载 demo 照跑。

## 一键复现

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q -W ignore::UserWarning        # 38 tests, 全绿
python examples/run_demo.py             # 端到端基准 -> examples/benchmark.json
python -m semiforge.cli run --datasets moons circle --hpo --hpo-trials 20 --hpo-timeout 120
```

Docker：

```bash
docker build -t semiforge .
docker run --rm semiforge python examples/run_demo.py
```

## 架构

```
semiforge/
  core/       types(SSLDataset/StrategyResult/BenchmarkRow) · errors(E100~E500)
              config(ENV_SEMIFORGE_* 覆盖) · interfaces(SSLStrategy Protocol)
  data/       synthetic(moons/blobs/circle/spiral/highdim) · loaders(digits 子采样)
  ssl/        estimators(后端工厂+numpy 兜底) · graph(kNN 传播)
              strategies(4 基础策略) · hybrid(CBCP-Hybrid 旗舰) · registry
  eval/       metrics(accuracy/macro_f1) · benchmark(runner/aggregate/table)
  hpo/        tune(Optuna TPE, timeout<=0 走兜底)
  pipeline/   SemiForgePipeline.run() -> JSON + 表格
  cli.py      `python -m semiforge.cli run ...`
```

调用单向无环：`cli → pipeline → {data, hpo, ssl, eval} → core`。
细节见 [docs/architecture.md](docs/architecture.md)。

## 质量

- **38 pytest 全绿**（单测含硬不变量：图传播行归一、钳位保真、混合策略
  必须在流形数据上显著超过监督基线等）
- **ruff check + format --check 全绿**（line-length 96）
- 确定性：全链路固定 seed，无并行竞争（RF `n_jobs=1`）
- License: MIT · Author: **晨星**
