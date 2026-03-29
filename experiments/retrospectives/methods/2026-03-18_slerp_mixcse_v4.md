# SlerpMixCSE（HardNegativeNLLLossV4）

## 背景
- 配置：`train_configs/slerp_interpolation/Llama31-8b-inst-mntp-supervised_SlerpMixCSE@DermVariantsSFT.json`
- 运行入口：`experiments/run_supervised_with_eval.py`
- 命令：`train_configs/slerp_interpolation/commands_slerpmixcse.sh`

## 方法意图
- 用球面插值替代线性插值，在正样本和 hardest negative 之间构造混合样本。
- 保持固定 lambda（`lam=0.2`），并为每一行追加一个专属 mixed negative。

## 可能失败的原因
- 几何层面的改进（Slerp）只有在 hard negative 本身信息有效时才有意义；它无法修正错误标注或过难负样本。
- 每行只加一个 mixed negative，在大候选池下影响可能有限。
- 对于难度结构差异很大的异构子集，固定 lambda 可能并不合适。

## 结论
- 保留为一个受控变体，不作为默认训练路径。

## 下一步检查
- [ ] 在除插值模式外完全相同的配置下比较 Lerp 与 Slerp。
- [ ] 看子集级收益，判断 Slerp 是否只对少数子集有帮助。
