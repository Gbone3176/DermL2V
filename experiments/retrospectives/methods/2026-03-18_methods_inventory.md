# 方法清单（LLM2Vec 回顾）

## 范围
这个索引用来跟踪仓库里已经落地过的方法级尝试，避免重复盲试，把迭代切换到基于证据的路线。

## 方法文件
- `2026-03-18_llm2vec_core_modifications.md`
- `2026-03-18_latent_pooling_designs.md`
- `2026-03-18_loss_versions_v0_to_v5.md`
- `2026-03-18_focalmixcse_v3.md`
- `2026-03-18_slerp_mixcse_v4.md`
- `2026-03-18_topk_shared_slerp_mixcse_v5.md`
- `2026-03-18_layer_fusion_pooling.md`
- `2026-03-18_training_free_token_pretend.md`

## 下一轮优先级
1. 固定一个可复现的基线（`HardNegativeNLLLoss` 或 V1 风格 MixCSE），并带上子集级指标重新跑一遍。
2. 在再次改模型或 loss 之前，先验证数据侧的 margin 质量。
3. 先做低成本消融（单 epoch 或限制步数），再决定是否启动任何 30 小时 full run。
