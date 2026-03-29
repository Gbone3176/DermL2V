# TopK 共享式 SlerpMixCSE（HardNegativeNLLLossV5）

## 背景
- 配置：`train_configs/topk_shared_slerp_mixcse/Llama31-8b-inst-mntp-supervised_TopKSharedSlerpMixCSE@DermVariantsSFT.json`
- 运行入口：`experiments/run_supervised_with_eval.py`
- 命令：`train_configs/topk_shared_slerp_mixcse/commands_topk_shared_slerpmixcse.sh`

## 方法意图
- 先按 V4 方式构造 mixed negatives，再在 batch 内共享，并为每个 query 选取 top-k（`shared_mix_topk`）。
- 目标是在不使用全部 mixed negatives 的前提下，丰富 hard distractor。

## 可能失败的原因
- 共享 top-k 可能引入跨样本污染：对某个 query 有意义的 hard negative，对另一个 query 可能只是语义不匹配的噪声。
- 从带噪 mixed pool 中选出的 top-k 可能加大 false-hard 压力。
- 在 mixed pool 上再做一次检索会增加复杂度，但并不能保证监督更干净。

## 结论
- 只有在有严格子集级诊断并且对 V4 做过消融对照之后，才适合继续使用。

## 下一步检查
- [ ] 在短程预算下扫一组较小的 `shared_mix_topk`。
- [ ] 衡量 top-k 共享负样本是否增加了 near-miss false negative。
