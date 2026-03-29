# Layer Fusion Pooling（LLM2VecV3）

## 背景
- 配置：`train_configs/Baseline_MixCSE_Fusion/MetaLlama3.1_8B_inst-mntp_supervised@DermVariantsSFT_mixcse_fusion.json`
- 运行入口：`experiments/run_supervised_fusion_withEval.py`
- 设计备注：`train_configs/Baseline_MixCSE_Fusion/FUSION_MODULE_NOTES.md`

## 方法意图
- 新的 `pooling_mode="layer_fusion"` 学习最后 K 层的加权组合。
- 引入 router / norm / gamma 参数，希望相对单层 pooling 提升表示质量。

## 可能失败的原因
- 如果主干大部分被冻结而 fusion head 又比较小，容量可能不足以修复数据层面的不一致。
- 如果同时训练 LoRA 和 fusion，两者的优化目标可能互相竞争。
- 更低的学习率和被改动的训练 schedule 可能会混淆收益或失败究竟来自 fusion 本身还是其他因素。

## 结论
- 保留为一个结构分支，但必须在相同优化预算下与 mean pooling 做对照。

## 下一步检查
- [ ] 做 `fusion-only` 与 `LoRA+fusion` 的受控对比。
- [ ] 记录各 step 学到的 layer weight，检查是否塌缩到单层。
