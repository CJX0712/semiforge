# SemiForge 踩坑与教训 (Pitfalls)

> 作者：晨星 (CJX0712) —— 本次构建过程真实踩过的坑，供复盘与复用。

## 1. 确定性杀手：`hash()` 受 PYTHONHASHSEED 影响
- **现象**：`make_semi_synthetic` 用 `SeedSequence([seed, ..., hash(dgp)%1000])` 派生种子，
  两次独立进程运行 benchmark **数字完全不同**（尽管单进程内 determinism_check 通过）。
- **根因**：Python 内置 `hash(str)` 默认受 `PYTHONHASHSEED` 随机盐化，跨进程不一致。
- **修复**：改用稳定编码 `int.from_bytes(dgp.encode("utf-8"), "little")`。
- **教训**：凡是参与随机种子派生的字符串，绝不能用内置 `hash()`；用 hashlib / from_bytes / 直接传字符串给 SeedSequence（其内部门禁用 hashlib 稳定哈希）。

## 2. RBF 全连接图在流形交错区失效
- **现象**：LabelSpreading 用 `kernel="rbf"` 在 moons 上交错区把不同簇点错误链接，精度仅 0.5–0.67。
- **修复**：改用 `kernel="knn"`（k=5），只连近邻，保留流形结构 → 稳定 0.75。

## 3. StandardScaler 破坏距离度量
- **现象**：对 20 维数据做 StandardScaler，冗余维度（原方差 0.01）被放大到方差 1，反客为主，
  主导欧氏距离，图方法与 LP 均退化。
- **修复**：图方法（LabelSpreading）与 LP-NumPy 均**直接使用原始特征**，让 inform 维主导距离。
  （仅 SupervisedOnly 的 LR 用 StandardScaler —— 那是它的正确用法。）

## 4. median heuristic 自适应 gamma
- **现象**：固定 `gamma=10` 在 20 维导致 RBF 亲和矩阵退化为近零，LP 收敛到 0.5。
- **修复**：`gamma = 1/(2*median(D2))`，对尺度自适应。

## 5. 对称归一化 + α=0.99 导致传播发散
- **现象**：用对称归一化 P 与 α=0.99（接近 1）迭代不收敛，`LabelPropagationNumpy` 长期不动。
- **修复**：改用**行归一化转移矩阵 W**（行和=1）+ 相同 α；收敛稳定，I2（单调非增）成立。

## 6. sklearn 1.9 把 `base_estimator` 改名为 `estimator`
- **现象**：`SelfTrainingClassifier(base_estimator=...)` 抛 `unexpected keyword`。
- **修复**：改用 `estimator=`（已兼容 1.9）。

## 7. 候选模型未在主 fit 路径 fit → NotFittedError
- **现象**：SemiFuse 主 fit 漏给候选 fit，预测时候选未训练。
- **修复**：fit 开头 `for c in self.candidates_: c.fit(X, y_labeled)`。

## 8. 少样本 OOF 路由失真
- **现象**：标注极少时，监督基线的 K-fold OOF 在折内标注全隐藏后只剩 1~2 个标注点，OOF 虚高
  （如 OOF=1.0 但真实 test=0.33），据此路由会误选。
- **修复**：废弃 OOF 决策，改为**标注预算自适应路由**（稀缺→图半监督，充足→纯监督）。

## 9. 结论聚合 bug：未限定方法过滤
- **现象**：`derive_failures` 计算 `fuse_a = mean(所有 abundant 行)`，误把全方法均值（0.746）
  当作 SemiFuse 均值，得出错误"diff=0.098 视为一致"。
- **修复**：显式过滤 `method=="SemiFuse"`；并统一以原始行值核对，避免聚合口径不一致。

## 10. "胜基线"必须是诚实的
- **现象**：最初期望 SemiFuse 全面碾压纯监督，但固定种子后发现正则化 LR 在 moons 上已 ~0.73，
  半监督并未全面胜出。
- **修复**：引入**强非线性监督基线**（XGB/RF 纯监督），诚实展示 SemiFuse 胜它们、与 LR 持平；
  并加入充足标注 regime 展示**诚实回退零负增益**。结论不夸大，反而更可信。

## 11. 环境：pip index / 429 资源包错误
- **现象**：`pip index versions` 触发 429 余额不足。
- **规避**：跳过 pip index，直接用已就绪的 numpy/scipy/sklearn/xgboost/optuna（preflight 已确认有 wheel）。
