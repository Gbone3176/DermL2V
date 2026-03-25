# Residual MLP Pooling Configs

本目录用于存放 `LLM2VecV4 + res_mlp_pooling` 的训练配置。

设计目的：

- 用轻量 residual MLP pooling 替代 NV-Embed 风格 latent pooling
- 检验“过强的随机 latent 结构会破坏原始 embedding 空间”这一假设
- 在尽量少改训练主流程的前提下，进行和 `mean` / `latent_pooling` 的直接对照

当前建议的第一轮对照：

- `mean`
- `latent_pooling`
- `res_mlp_pooling`

当前这组配置的默认约束策略：

- `res_mlp_num_layers = 4`
- `res_mlp_gamma_init = 1e-3`
- `res_mlp_output_normalize = true`
- `res_mlp_output_layernorm = true`
- `loss_class = HardNegativeNLLLossV5`
- `loss_kwargs.shared_mix_topk = 16`

这样既保留低扰动初始化，又延续之前训练里对 embedding 长度信息的抑制策略。

注意：

- 这些配置文件已经准备好字段，但训练脚本还需要显式接入 `llm2vecV4.py` 和 `pooling_mode="res_mlp_pooling"`，否则仅有配置还不能实际切换到 V4。
