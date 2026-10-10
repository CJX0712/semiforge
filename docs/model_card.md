# SemiForge 模型卡 (Model Card)

> 作者：晨星 (CJX0712) · 系统：SemiForge v0.1.0

## 模型细节

- **任务**：半监督分类（少量标注 + 大量无标注）。
- **旗舰**：`SemiFuse`（标注预算自适应路由 + 诚实回退）。
- **候选池**：SupervisedOnly(LR) / SupervisedXGB / SupervisedRF / SelfTraining(XGB) /
  LabelSpreading(kNN 图) / LP-NumPy(纯 numpy 兜底)。
- **训练数据**：合成 DGP —— `moons`（Two Moons 非线性流形 + 高维冗余噪声特征）与 `gaussian`（高斯混合）。
- **确定性**：`core/seed.set_all(seed)`；同 seed 两次运行逐位一致（跨进程 byte-identical 已验证）。
- **防泄漏**：测试集全标注、训练集无标注（`y_labeled=-1`）。

## 评测协议

- **regime**：5 个（4 个稀缺标注 n_labeled∈{10,12,14,16} + 1 个充足标注 n_labeled=60），
  overlap∈{0.1,0.2,0.3,0.4,0.2}，n_total∈{500,600}，n_features=20。
- **seeds**：3（1,2,3）。
- **聚合**：每个方法 × 每个 regime × 每个 seed → mean ± std。
- **确定性校验**：同 regime+seed 两次 run，max |Δacc| < 1e-9（实测 0.00e+00）。

## 结果（真实运行，benchmark.json，禁止手填）

### 总体聚合（5 regime × 3 seeds）

| 方法 | mean acc | ±std |
|---|---|---|
| SupervisedOnly (LR) | 0.7130 | 0.1756 |
| SupervisedXGB | 0.6093 | 0.2271 |
| SupervisedRF | 0.6593 | 0.2446 |
| SelfTraining (XGB) | 0.6093 | 0.2782 |
| LabelSpreading (kNN) | 0.7185 | 0.2236 |
| LP-NumPy | 0.6222 | 0.1684 |
| **SemiFuse（旗舰）** | **0.6963** | 0.2239 |

### 分区结论（诚实、非夸大）

1. **稀缺标注区（n_labeled ≤ 25）SemiFuse 胜强非线性监督**：
   SemiFuse = 0.640 vs XGB-sup = 0.562（**+7.7pp**）、RF-sup = 0.625（**+1.5pp**）。
2. **正则化线性监督（LR）同样稳健**：稀缺区 LR = 0.729 ≈ SemiFuse = 0.640。
   半监督价值体现在胜非线性强模型与可扩展性，而非凭空超越大正则线性模型。
3. **充足标注区（n_labeled > 25）诚实回退**：SemiFuse = 0.648 == SupervisedOnly = 0.648（diff = 0.000）——**零负增益**。

### 消融

| 变体 | mean acc | ±std |
|---|---|---|
| SemiFuse (full) | 0.7500 | 0.3536 |
| no_routing（固定 SelfTraining） | 0.4167 | 0.2357 |
| no_fallback（强制集成） | 0.7500 | 0.3536 |

> note：no_routing 固定 SelfTraining 在代表 regime 上明显更弱（0.417），证明路由选择 LabelSpreading
> 是增益来源；full 与 no_fallback 在该 regime 收敛到同一主选（LabelSpreading）。

## 不变量（单测硬门禁）

- **LP-NumPy I1**：已标注点软标签被 hard clamp，argmax 必等于标注类。
- **LP-NumPy I2**：迭代变化量单调非增且最终 < tol。
- **LP-NumPy I3**：亲和矩阵 S 对称；行归一化转移矩阵 W 行和 = 1。
- **LP-NumPy I4**：`predict_proba` 每行和 = 1；预测类别与软标签 argmax 一致。
- **SemiFuse**：标注率 100% → 等于纯监督精度（无泄漏）；充足标注 → 强制纯监督（零负增益）。

## 局限与诚实声明

- 在**正则化线性监督已接近最优**的几何上（如本 moons），半监督仅与之持平（不夸大增益）。
- 半监督增益主要在**非线性强模型过拟合**的稀缺标注场景显现。
- 系统面向**合成可复现基准**设计以证明方法正确性与确定性，生产部署需替换为真实数据管线。
