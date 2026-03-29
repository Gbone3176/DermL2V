# Loss 各版本回顾（V0 到 V5）

## 背景
- 注册表：`llm2vec/loss/utils.py`
- 实现：
  - `HardNegativeNLLLossV0.py`
  - `HardNegativeNLLLossV1.py`
  - `HardNegativeNLLLossV2.py`
  - `HardNegativeNLLLossV3.py`
  - `HardNegativeNLLLossV4.py`
  - `HardNegativeNLLLossV5.py`

## 版本摘要
- V0：基础 in-batch + 显式负样本交叉熵。
- V1：MixCSE 风格的固定 lambda 混合 hard negative。
- V2：V1 + focal reweighting（`gamma`）。
- V3：动态 lambda + margin-aware mixed-negative penalty。
- V4：固定 lambda 的 Slerp / Lerp 混合负样本。
- V5：V4 + batch 内共享的 top-k 混合负样本。

## 为什么这么多变体仍然可能失败
- 大多数变体仍然依赖 hardest-negative mining 的质量；如果挖出的负样本本身带噪，所有版本都会继承这种不稳定。
- 更复杂的 loss 会引入更多超参数，在固定算力预算下增加调参成本和结果方差。
- 如果不同子集的目标本身冲突（数据分析里已经观察到），仅靠 loss 微调无法修复标签空间不一致。

## 结论
- 保留这条 loss 家族，但在没有低成本筛选之前，停止直接做 full-run 超参扫。

## 下一步检查
- [ ] 先搭一个小步数基准（例如前 300 到 500 step）做 loss 排名，再决定是否跑完整 30 小时。
- [ ] 在每个 loss 版本下记录按子集划分的 margin 统计。
- [ ] 冻结一个稳定基线，用于后续回归测试。
