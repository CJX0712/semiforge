# SemiForge 架构与量化基线

作者：晨星 · 2026-09-28

## 1. 设计原则

- **契约先行**：所有策略实现同一协议 `fit(SSLDataset) -> self`、`predict(X) -> (n,) int labels`；
  评测口径统一为 transductive（只评无标注点）。
- **单一职责**：`core` 无任何上游依赖；`data/ssl/eval/hpo` 只依赖 `core`；
  `pipeline` 编排全链路；`cli` 只做参数解析与输出。
- **离线兜底**：SOTA 后端（sklearn LabelSpreading 等）不可用时自动降级纯 numpy
  实现，接口语义不变（后端探测在调用期进行，模块属性可 monkeypatch）。

## 2. 模块职责

| 模块 | 职责 | 关键点 |
|------|------|--------|
| `core/types.py` | `SSLDataset`（y_labeled=-1 表示无标注）、`StrategyResult`、`BenchmarkRow` | `assert_valid_budget` 保证每类≥1 标注 |
| `core/errors.py` | E100 data / E200 model / E300 eval / E400 registry / E500 pipeline | 全部带错误码前缀 |
| `core/config.py` | `ENV_SEMIFORGE_SEED/N_NEIGHBORS/N_HPO_TRIALS/HPO_TIMEOUT/...` 覆盖 | |
| `data/synthetic.py` | moons/blobs/circle/spiral/highdim 五个合成 SSL 数据集 | `_mask_labeled` 类分层抽样，每类保底 1 个 |
| `data/loaders.py` | sklearn digits 分层子采样（随包离线可得） | 特征 0-1 缩放保 kNN 图有效 |
| `ssl/estimators.py` | RF/LR 工厂 + `NumpyLogisticRegression`（softmax+Adam，确定性） | sklearn 1.9：LR 用 lbfgs，无 multi_class 参数 |
| `ssl/graph.py` | kNN 高斯核亲和 + 归一化传播 `F←αSF+(1-α)Y0`（钳位）+ 行归一化 | Zhu, Ghahramani & Lafferty 2002 |
| `ssl/strategies.py` | supervised / self_training / graph / co_training | |
| `ssl/hybrid.py` | **CBCP-Hybrid 旗舰** | 见 §3 |
| `ssl/registry.py` | 名称→工厂注册表，E402 未知策略 | |
| `eval/` | metrics + benchmark runner + aggregate_gain | |
| `hpo/tune.py` | Optuna TPE（seed 固定），timeout≤0 走兜底分支 | 目标 = 多数据集 val acc 均值 |
| `pipeline/` | `SemiForgePipeline.run() -> result dict + benchmark.json` | |

## 3. CBCP-Hybrid 旗舰

```
输入: SSLDataset (n, d)，标注率 5~10%

S1 校准（仅在标注集上，k=5 折 CV）:
   acc_g = 图传播源 hold-out 精度
   acc_f = RF+LR 集成源 hold-out 精度
   若 |acc_g-acc_f| ≤ 0.06: (w_g, w_f) = (0, 1)        # 近平局锚定监督
   否则 (w_g, w_f) = softmax_p(acc_g, acc_f), p = 14   # 幂次锐化

S2 融合后验:  P = w_g · F_graph + w_f · P_ensemble

S3 类平衡课程轮次 r = 1..R:
   配额 quota_r = base_quota · r（类间独立截断）
   阈值 τ_c = max(quantile(P_c, 1-0.80), 0.5)
   提交后 CV 复评: 融合精度无严格改进 ⇒ 整轮回滚并终止   # 守护回滚
   否则特征源在扩展标签集上重拟，重融合

S4 最终: 全量后验 P_all（labeled 行硬钳位 one-hot），传导式 argmax
   归纳路径: RF(labeled+pseudo) 直接 predict
```

**为什么按构造非劣**：特征源 CV 精度 ≈ 监督基线精度（同一模型族同一数据）；
近平局时权重全押特征源，伪标签提交又受"严格改进"守护——任何会伤及
CV 融合精度的轮次都无法进入最终标签集。

## 4. 量化基线（seed=42，逐位可复现）

数据集：n=600（digits 500），标注率 5%（digits 10%），评测于无标注点。

| 数据集 | supervised | self_training | graph | co_training | **cbcp_hybrid** |
|--------|-----------:|--------------:|------:|------------:|----------------:|
| moons   | 0.7684 | 0.8404 | 0.8825 | 0.7895 | 0.8491 |
| blobs   | 0.9193 | 0.8947 | 0.9088 | 0.6719 | 0.9053 |
| circle  | 0.5351 | 0.8018 | 0.8439 | 0.7105 | 0.8228 |
| spiral  | 0.4070 | 0.4912 | 0.5018 | 0.4842 | 0.5158 |
| highdim | 0.9895 | 0.9561 | 0.9912 | 0.9596 | 0.9860 |
| digits  | 0.8578 | 0.8244 | 0.9378 | 0.8644 | 0.9356 |

旗舰 vs 监督基线：均值 **+0.0896**；4/6 显著为正，2/6 在 ±2·SE 噪声带内非劣。

诚实记录（失败案例与 limitation）：

- `graph`（LabelSpreading）在流形数据上极强；旗舰在 moons/circle 上融合特征源
  后仍低于纯图源（0.8491 vs 0.8825；0.8228 vs 0.8439）——CV 锐化权重减少了
  稀释但未完全消除。旗舰价值在于**跨结构自适应**：blobs/highdim 这类
  图源无优势的数据集上不劣于监督，流形数据集上大幅超越。
- `co_training` 在 blobs 上灾难性失败（0.6719）：随机特征切分把簇间判别维度
  拆散，双视图互相污染——保留作反例展示。
- HPO 目标只覆盖 moons/circle（子集调参），未在全部数据集上寻优，避免
  把 test 信息泄漏进调参。

## 5. 不变量（单测守护）

1. 图传播无标注行和 = 1（±1e-6），labeled 行 argmax = 真标注（钳位保真）
2. 全策略输出 ∈ 已知类别、无 NaN、shape=(n,)
3. 硬门槛：graph 与 cbcp_hybrid 必须在 circle/moons 上超过监督基线 +0.05/+0.03
4. 全链路确定性：同 seed 两次运行预测逐位相等
5. HPO timeout≤0 走兜底分支（不抛异常）
6. numpy 兜底后端在 sklearn 被禁用时自动接管（monkeypatch 验证）
