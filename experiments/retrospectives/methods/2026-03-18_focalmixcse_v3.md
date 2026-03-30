# FocalMixCSE（HardNegativeNLLLossV3）

## 背景
- 配置：
  - `train_configs/FocalMixCSE/Llama31-8b-inst-mntp-supervised_FocalMixCSE@DermVariantsSFT.json`
  - `train_configs/FocalMixCSE/Llama31-8b-inst-mntp-supervised_MixCSE@DermVariantsSFT.json`
- 运行入口：`experiments/run_supervised_FocalMixCSE.py`

## 方法意图
- 使用动态 lambda 和带 margin 感知的混合负样本惩罚，把训练重点放到更难的样本上。
- 在 focal 变体中，`mix_weight=1.0`；在 V3 下的基线配置中，`mix_weight=0.0` 作为对照。

## 可能失败的原因
- 动态 lambda 由当前 hard-negative 与 positive 的分数关系驱动；如果负样本本身噪声较大，控制信号也会随之变噪。
- margin 和 focal 效应可能会过度放大本来就模糊的样本，尤其是边界样本或噪声负样本。
- 方法复杂度更高后，容易掩盖真正的瓶颈其实来自数据组成。

## 结论
- 只有在按子集验证过负样本质量之后，才适合继续使用这条方法线。

## 下一步检查
- [ ] 训练过程中按子集绘制 `s_pos - s_hard` 的分布。
- [ ] 在相同随机种子和短程预算下比较 V3 中 `mix_weight=0` 与 `1`。
