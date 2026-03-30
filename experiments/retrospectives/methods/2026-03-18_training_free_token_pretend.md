# 免训练 Token Pretend / Separator-Mask 方法

## 背景
- 主代码路径：
  - `llm2vec/llm2vecV1.py`（`tokenize_with_separator`、`embed_mask`、`_skip_instruction`）
  - `experiments/run_supervised_with_eval.py`（`_encode_texts` 要求存在 separator）
- 常用 separator：`!@#$%^&*()`

## 方法意图
- 不改模型权重，仅通过 separator 和 `embed_mask` 把内容 token 与 instruction token 隔离开。
- 之后的 pooling 只聚合内容区域，减少 instruction 泄漏。

## 可能失败的原因
- 强依赖格式：一旦缺少 separator，会直接失败或生成退化 mask。
- 双 tokenization 路径（`original_texts` 与 content-only ids）在截断和 chat template 场景下比较脆弱。
- 如果 query / document 的 prompt 结构在不同子集里不一致，mask 逻辑可能对不准真正的语义目标。
- 免训练 masking 无法解决数据中的标签噪声或 hard-negative 歧义。

## 需要跟踪的关键失败信号
- mask 后有效内容长度为零或接近零。
- 各子集间 masked length 分布漂移。
- 性能对 separator 放置位置和文本预处理过于敏感。

## 结论
- 这是一个有用的工程技巧，但不是当前性能不足的核心解法。

## 下一步检查
- [ ] 增加运行时统计：每个 batch / 子集的 mask 覆盖率。
- [ ] 在 dataset loader 层验证 separator 的存在和位置。
- [ ] 做一个小型消融：在相同 checkpoint 下比较 `skip_instruction=True` 与 `False`。
