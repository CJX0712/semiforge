# SemiForge 架构设计

> 作者：晨星 (CJX0712)

## 1. 设计原则

1. **数学/算法正确性优先**：每个模块都有可被独立验证的「不变量」，并配单测硬门禁。
2. **确定性可复现**：同 seed 两次运行核心指标逐位一致（已验证跨进程 byte-identical）。
3. **防数据泄漏**：测试集全标注、训练集无标注（`y_labeled=-1`），无标注信息永不进入训练。
4. **诚实降级**：标注充足时不编造增益，强制回退纯监督；绝不输出负增益。
5. **复用世界顶级开源 + 零依赖兜底**：Tier-0 用 scikit-learn / XGBoost；Tier-1 纯 numpy 兜底永可运行。

## 2. 调用链（单向无环）

```
pipeline ──► data ──► semi ──► eval
    │          │        │        │
    └──────────┴────────┴────────┴──► core (底座: seed/config/types/errors)
```

`pipeline` 编排训练/评测/聚合；`data` 生成与切分；`semi` 实现模型；`eval` 计算指标；
所有模块最终依赖 `core`。

## 3. 模块职责

### core/
- `seed.py`：`set_all(seed)` 一次性设齐 `random` / `numpy`；`rng()` 返回确定性 Generator。
- `config.py`：`Config` dataclass + `get_config()`，支持 `SEMIFORGE_` 环境变量覆盖与 schema 校验。
- `types.py`：`SemiData` / `BenchmarkRow` / `BenchmarkResult` / `AblationResult` 数据结构。
- `errors.py`：错误码 E100–E500（ConfigError/DataError/ModelError/EvalError/...）。
- `interfaces.py`：`SemiSupervisedModel` Protocol（fit/predict/predict_proba/score）。

### data/
- `synthetic.py`：两种 DGP —— `gaussian`（高斯混合）与 `moons`（Two Moons 非线性流形 +
  高维冗余噪声特征）。关键：`make_semi_synthetic` 用 `SeedSequence([seed, n_features, n_classes, dgp_salt])`
  派生独立 Generator，**不使用受 `PYTHONHASHSEED` 影响的 `hash()`**，保证跨进程可复现。
  不变量：① `sum(y_labeled==-1)==n_total-n_labeled`；② 每类至少 `min_per_class` 标注；③ 类覆盖完整。
- `loaders.py`：`train_test_split_semi` —— 测试集全标注、训练集无标注、按类分层抽样；
  `load_semi_npz` / `save_semi_npz` 本地持久化。

### semi/
- `base.py`：复用世界顶级开源的封装：
  - `SupervisedOnly`：LogisticRegression + StandardScaler（纯监督最强基线）。
  - `SupervisedXGB` / `SupervisedRF`：纯监督强非线性基线（XGB 不可用时自动降 RF）。
  - `SelfTrainingXGB`：SelfTrainingClassifier(XGB)。
  - `LabelSpreadingSK`：LabelSpreading **kNN 图**（非 RBF 全连接）。
  - `available_xgboost()`：探测降级。
- `lp_numpy.py`：**纯 numpy 迭代式标签传播**（Zhou 2004 / Bengio 2006）。RBF 亲和 + 行归一化转移矩阵
  + hard clamping；median heuristic 自适应 gamma；不做 StandardScaler。不变量 I1–I4（见 model_card）。
- `semifuse.py`：旗舰 `SemiFuse` —— 标注预算自适应路由 + 诚实回退。

### eval/
- `metrics.py`：手写 `accuracy_score`（与 sklearn 一致性已验证）、`mean_std`。

### pipeline/
- `pipeline.py`：`SemiPipeline.run`（单数据全方法评测）+ `benchmark`（多 regime × 多 seed 聚合）。

## 4. SemiFuse 路由逻辑

```
fit(X, y_labeled):
  1. 候选池全部 fit (SupervisedOnly/SelfTraining/LabelSpreading/LP-NumPy)
  2. if 全部标注 (np.all(y_labeled!=-1)): 强制纯监督
  3. if 标注稀缺 (n_labeled <= label_budget): 选 LabelSpreading (kNN 图)
  4. if 标注充足 (n_labeled >  label_budget) & use_fallback: 强制 SupervisedOnly
```

消融开关：`use_routing=False` → 固定 SelfTraining（量化路由增益）；
`use_fallback=False` → 走稀缺路由（量化回退价值）。

## 5. 复现保证

- 数据：`make_semi_synthetic(seed=s)` 与 `train_test_split_semi(seed=固定)` 各自派生独立 RNG，
  与全局状态隔离；不使用 `hash()`。
- 模型：XGBoost / LabelSpreading / LogisticRegression 均传入固定 `random_state`；
  LP-NumPy 纯 numpy 无 RNG。
- 验证：`determinism_check` 与 CI 的 demo 重跑均确认 accuracy 逐位一致。
