# 残差 MLP Pooling 假设（V4）

## 背景

- 动机：解释为什么 NV-Embed 风格的 latent pooling 在当前皮肤科微调流程中带来了负收益。
- 新代码：
  - `llm2vec/llm2vecV4.py`
  - `llm2vec/pooling_residual_mlp.py`

## 工作假设

当前的基础编码器已经具备一个还不错的自然语言嵌入空间。
在这种前提下，NV 风格 latent pooling 可能有害，因为：

- 它增加了相对较多的额外参数
- 它引入了随机初始化的 latent prototype
- token 到 latent 的 cross attention 会在训练初期注入噪声
- 早期优化可能会严重扭曲原有的嵌入几何

结果就是，模型可能因为表示结构被破坏而损失更多，而不是从额外适配能力中得到更多收益。

## V4 设计

用一个轻量级残差 MLP pooler 替换 latent pooling：

- token 表示保持在原始 hidden space 中
- 一个 4 层 MLP 只预测残差修正项
- 残差分支由一个较小的 `gamma` 缩放
- 最后一层线性层做零初始化

这样初始化会非常接近普通 mean pooling，因此领域适配从“小修正”开始，而不是从“大改写”开始。

## 为什么这是一个好测试

如果 V4 比 latent pooling 更好，那么更可能的问题并不是“pooling 需要更大容量”，而是：

- 适配分支太有破坏性
- 随机 latent prototype 的交互噪声太大
- 保留预训练嵌入几何比增加复杂 pooling 结构更重要

如果 V4 仍然以同样方式失败，那么瓶颈更可能在别处：

- 数据混合冲突
- hard negative 质量
- loss 不稳定
- 优化 schedule

## 推荐的第一组比较

在以下条件完全相同的前提下做一个短程受控对比：

- base model
- dataset mix
- seed
- batch size
- learning rate
- loss

比较：

1. `mean`
2. `latent_pooling`
3. `res_mlp_pooling`

## 需要重点观察的指标

- 验证集检索指标
- 前 100 到 500 step 的训练稳定性
- `grad_norm`
- 验证集下滑是否出现得更早或更严重
- 相对基线编码器的 cosine similarity 分布漂移

## 值得尝试的额外消融

- V4 搭配 `gamma_init=1e-3`
- V4 搭配 `gamma_init=1e-2`
- V4 是否使用输出 L2 normalization
- V4 是否在 pooled output 后加 LayerNorm

## 判定规则

- 如果 V4 > latent pooling，并且接近或超过 mean pooling：
  那么“latent 结构过于有破坏性”这个假设会更可信。
- 如果 V4 < mean pooling，但 > latent pooling：
  说明轻量修正是有帮助的，但主要瓶颈可能仍在数据或 loss。
- 如果 V4 和 latent pooling 都不如 mean pooling：
  说明现阶段增加 pooling 容量可能不是正确杠杆。
