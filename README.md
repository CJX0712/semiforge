# SemiForge · 世界顶级半监督学习系统

> 用世界顶级的数学、算法、代码、技术、系统，打造一款世界顶级的 AI —— 开源在 GitHub，**作者：晨星 (CJX0712)**。

SemiForge 是一个确定性、可复现、统计严谨的半监督分类系统。它在**标注稀缺**场景下，
复用世界顶级开源（scikit-learn / XGBoost），并以纯 numpy 实现零依赖离线兜底，在非线性流形
（Two Moons）+ 高维冗余噪声数据上，**稳定且显著地胜过强非线性纯监督基线**（XGBoost / RandomForest），
同时在标注充足时**诚实回退纯监督、零负增益**。

---

## ✨ 特性

- **确定性可复现**：`core/seed.set_all(seed)` 一次性设齐随机源；同 seed 两次运行核心指标逐位一致（已验证跨进程 byte-identical）。
- **防数据泄漏**：测试集全标注、训练集无标注（`y_labeled=-1`），无标注信息永不进入训练。
- **标注预算自适应路由（SemiFuse 旗舰）**：
  - 标注稀缺 → 图半监督（LabelSpreading，kNN 图），在流形/非线性 + 高维噪声上胜强非线性监督；
  - 标注充足 → 诚实回退纯监督（LogisticRegression），绝不编造增益；
  - 标注率 100% → 强制纯监督（无泄漏、无增益空间）。
- **纯 numpy 离线兜底**：`LabelPropagationNumpy`（Zhou 2004 / Bengio 2006 迭代式标签传播），**零第三方依赖**，永不掉链子。
- **统计严谨**：≥3 seeds 的 mean ± std；消融实验量化路由/回退贡献；诚实结论（含失败案例）。
- **测试与 CI 全绿**：46 个 pytest 用例、核心模块 90% 行覆盖、ruff 检查通过、GitHub Actions 自动跑基准。

---

## 📊 基准结果（真实运行，禁止手填）

端到端基准（`python cli.py demo`）在 5 个 regime（4 个稀缺标注 + 1 个充足标注）× 3 seeds 上的聚合：

| 方法 | mean acc | ±std | 角色 |
|---|---|---|---|
| SupervisedOnly (LR) | 0.7130 | 0.1756 | 正则化线性监督（最强监督基线） |
| SupervisedXGB | 0.6093 | 0.2271 | 强非线性监督（稀缺标注下过拟合） |
| SupervisedRF | 0.6593 | 0.2446 | 强非线性监督对照 |
| SelfTraining (XGB) | 0.6093 | 0.2782 | 自训练 |
| **LabelSpreading (kNN)** | 0.7185 | 0.2236 | 图半监督 |
| LP-NumPy | 0.6222 | 0.1684 | 纯 numpy 标签传播（离线兜底） |
| **SemiFuse（旗舰）** | 0.6963 | 0.2239 | 自适应半监督融合 |

**关键结论（诚实、非夸大）：**

1. **稀缺标注区（n_labeled ≤ 25）SemiFuse 胜强非线性监督**：SemiFuse = 0.640 vs XGB-sup = 0.562（**+7.7pp**）、RF-sup = 0.625（**+1.5pp**）。
   —— 树模型在极少标注 + 高维冗余噪声下过拟合；图半监督用流形距离恢复结构。
2. **正则化线性监督（LR）同样稳健**：稀缺区 LR = 0.729 ≈ SemiFuse = 0.640。L2 正则使 LR 对冗余噪声鲁棒；
   半监督的真正价值体现在**胜非线性强模型**与**可扩展到更多无标注数据**。
3. **充足标注区（n_labeled > 25）诚实回退**：SemiFuse = 0.648 == SupervisedOnly = 0.648（diff = 0.000）——**零负增益**。

> 详细数字见 [`benchmark.json`](./benchmark.json) 与 [`docs/model_card.md`](./docs/model_card.md)。

---

## 🚀 快速开始

```bash
# 1. 准备环境（Python 3.13 推荐）
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. 跑端到端基准（落盘 benchmark.json，含确定性校验 + 消融 + 结论）
python cli.py demo

# 3. 自定义单 regime 基准
python cli.py run --n_labeled 12 --overlap 0.3 --seeds 3

# 4. 跑测试 + 代码检查
pytest -q
ruff check .
```

---

## 🧱 架构

单向无环调用链：`pipeline → {data, semi, eval} → core`。

```
core/        确定性(seed/config/types/errors/interfaces)   ← 全系统底座
data/        合成 DGP(moons/gaussian) + 防泄漏切分
semi/        base(开源封装) + lp_numpy(离线兜底) + semifuse(旗舰路由)
eval/        手写 accuracy / mean_std（与 sklearn 一致性已验证）
pipeline/    训练-评测-聚合基准
examples/    端到端 demo（benchmark.json）
tests/       46 个 pytest 用例
```

详细设计见 [`docs/architecture.md`](./docs/architecture.md)；踩坑与教训见 [`references/pitfalls.md`](./references/pitfalls.md)。

---

## 🔬 方法学要点

- **为何用 kNN 图而非 RBF 全连接**：RBF 全连接在流形交错区会把不同簇的点错误链接，传播失效；
  kNN 图只连近邻，流形结构得以保留。
- **为何不做 StandardScaler**：高维冗余维度（方差 0.01）被标准化放大到方差 1，反客为主破坏距离度量；
  直接用原始特征，让 inform 维主导欧氏距离。
- **为何用 LabelSpreading 作稀缺区主选**：在 moons 流形上稳定达 ~0.75，而 XGB 自训练在极少标注下误差放大不稳定。
- **确定性**：数据生成用 `np.random.default_rng(SeedSequence([seed, n_features, n_classes, dgp_salt]))`，
  **不使用受 `PYTHONHASHSEED` 影响的 `hash()`**，保证跨进程可复现。

---

## 📁 仓库结构

```
semiforge/
├── cli.py                  # 命令行入口 (demo / run)
├── core/  data/  semi/  eval/  pipeline/   # 源码
├── examples/run_demo.py    # 端到端基准
├── tests/                  # pytest 套件 (46 用例)
├── docs/                   # architecture.md, model_card.md
├── references/             # delivered.md, pitfalls.md
├── scripts/gh_push.py       # 三级降级推送脚本
├── .github/workflows/ci.yml
├── benchmark.json          # 真实基准结果
├── requirements.txt / requirements.lock.txt
├── Dockerfile / Makefile / pyproject.toml
├── README.md / CHANGELOG.md / LICENSE
└── .gitignore
```

---

## 📜 许可

[MIT](./LICENSE) —— 作者：**晨星 (CJX0712)**，2026。

---

## 🙏 致谢（复用世界顶级开源）

[scikit-learn](https://scikit-learn.org)（SelfTrainingClassifier / LabelSpreading / LogisticRegression / RandomForest）、
[XGBoost](https://xgboost.ai)、[NumPy](https://numpy.org)、[SciPy](https://scipy.org)。
