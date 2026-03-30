# Latent Pooling 设计演进（V0/V1/V2/V3）

## 背景
- 文件：
  - `llm2vec/pooling_latent_V0.py`
  - `llm2vec/pooling_latent_V1.py`
  - `llm2vec/pooling_latent_V2.py`
  - `llm2vec/pooling_latent_V3.py`
  - `llm2vec/pooling_latent.py`

## 设计演进
- V0：基础 latent dictionary + MHA + MLP + mean pool。
- V1：加入 LayerNorm 和残差路径。
- V2：采用 NV-Embed 风格的 PreNorm + cross-attn + FFN + residual。
- V3：在 V2 基础上增加对 SDPA / flash 后端的兼容，以及 dropout 细化。
- `pooling_latent.py`：实际运行分支，包含更安全的初始化和对 DDP 更稳妥的 device cast。

## 潜在失败原因
- 目标不匹配：latent pooling 增加的是表达能力，但如果真正瓶颈是 noisy hard negative 或数据冲突，增加容量未必有帮助。
- 优化负担：额外的 pooling 参数在高学习率或大有效 batch 下可能会让前期训练更不稳定。
- 接口混淆：实验可能以为自己在用 V2 / V3，但运行时其实通过当前 import 链路用的是 `pooling_latent.py`。

## 风险备注
- 早期的 V0 / V1 模式在 forward 中包含 `self.to(device)` 这类设备迁移，在分布式训练里比较脆弱。
- latent 模块只有在注意力 mask 语义正确且一致时才可能有效，例如 `embed_mask` 与 `attention_mask` 的定义必须一致。

## 结论
- 将 latent pooling 视为次级杠杆；优先处理数据和 loss 的诊断问题。

## 下一步检查
- [ ] 显式打印运行时真正使用的 latent pooling 模块文件与版本。
- [ ] 在相同随机种子、相同数据顺序和短预算下比较 `mean` 与 `latent_pooling`。
