# HardNegativeNLLLoss 各版本相对 V0 的增量说明

本文档以 `HardNegativeNLLLossV0.py` 为基线，概括 `llm2vec/loss/` 目录下各个相关损失函数版本相对于 V0 的主要修改点。

## V0 基线行为

`HardNegativeNLLLossV0.py` 的核心逻辑是：

- 将 `d_reps_pos` 和 `d_reps_neg` 直接拼成一个全局候选池。
- 对每个 query，正例标签固定为对角线位置，也就是 `q_i -> pos_i`。
- 所有显式 negative 都会被当作“所有 query 的共享负例”。
- loss 形式是标准 `CrossEntropyLoss`。

可把它理解为：

```text
候选列 = [全体 positive | 全体 explicit negative]
监督方式 = 单正例 CE
```

它的优点是简单直接，但默认假设比较强：

- 显式 negative 不会和任何 positive 重复。
- 某个样本的 hard negative 也适合作为其他样本的负例。
- 只需要一个主正例，不考虑 multi-positive。

## 各版本相对 V0 的增量

### `HardNegativeNLLLossV0_1.py`

核心增量：

- 新增“显式 negative 与某个 positive 完全相同”的检测。
- 如果某个 negative 实际和某个 positive 向量完全一致，就把该 logit 直接 `mask` 掉。

相对 V0 的变化：

- V0：重复样本仍会被当负例参与 softmax。
- V0_1：重复样本不再当负例，但也不提升为正例，只是从候选里删掉。

影响：

- 缓解最明显的 false negative / duplicate negative 问题。
- 仍然保持单正例 CE，不改变正例定义。

一句话总结：

```text
V0_1 = V0 + “把和 positive 完全重复的 negative 从 softmax 中移除”
```

### `HardNegativeNLLLossV0_2.py`

核心增量：

- 不再把 `d_reps_neg` 视为所有 query 共享的全局负例池。
- 强制 `d_reps_neg` 与 `q_reps` 按行对齐，即 `neg_i` 只服务于 `query_i`。
- 每个 query 只额外看到自己那一个 negative logit。

相对 V0 的变化：

- V0：`neg_k` 会被所有 query 一起拿来做负例。
- V0_2：`neg_i` 只参与第 `i` 行 loss，避免“别人的 hard negative 被误用成我的负例”。
- 同时保留 in-batch positive 对比，即 `logits_pos` 仍是 `[B, B]`。

额外处理：

- 如果 `neg_i == pos_i`，会把这一行的专属 negative mask 掉。

影响：

- 更适合“每个 query 都有自己专属 hard negative”的数据组织方式。
- 明确减少跨样本负迁移。

一句话总结：

```text
V0_2 = V0 + “explicit negative 改为按行专属，而不是全局共享”
```

**实验证明, 在 V0_2 的损失函数的效果相对更好**

### `HardNegativeNLLLossV0_3.py`

核心增量：

- 和 V0_1 一样先检测“negative 是否其实和某个 positive 完全相同”。
- 但处理策略不是删掉，而是把这些列并入正类集合。
- loss 从单正例 CE 改成 multi-positive NLL。

相对 V0 的变化：

- V0：每行只有一个正例，对角线之外都视为负例。
- V0_3：若某些 negative 与 `pos_i` 完全相同，那么这些列也算 `query_i` 的正例。

数学形式变化：

- 分子不再是单个正例 logit，而是“所有正例列的 `logsumexp`”。
- 分母仍是全部候选列的 `logsumexp`。

影响：

- 比 V0_1 更积极地利用重复样本信息。
- 适合“同义、重复、去重不充分”的场景。

一句话总结：

```text
V0_3 = V0 + “false negatives 不删除，而是提升为额外正例，改用 multi-positive NLL”
```

### `HardNegativeNLLLossV0StructuredSelfAttnAblation.py`

核心增量：

- 保持 V0 的检索 loss 主体不变。
- 额外支持 `aux_loss` 和 `aux_loss_weight`。

相对 V0 的变化：

- retrieval 部分与 V0 基本一致。
- 最终 loss 变为：

```text
retrieval_loss + aux_loss_weight * aux_loss
```

用途：

- 主要用于 structured self-attention 的消融，不改动原始 V0 本体。

一句话总结：

```text
V0StructuredSelfAttnAblation = V0 + 可选辅助正则项
```

### `HardNegativeNLLLoss.py`

这个文件本质上是一个 MixCSE 风格版本，不再是纯 V0 思路。

核心增量：

- 在 V0 的 `[positive | explicit negative]` 候选基础上，新增一列“mixed negative”。
- mixed negative 来自：
  `mixed_i = lam * pos_i + (1 - lam) * hard_neg_i`
- 其中 `hard_neg_i` 是从显式 negative 池里为 `query_i` 挖到的最难负例。
- mixed negative 是“按行单独追加的一列”，不是共享给所有 query 的全局候选。
- mixed 分支默认 `detach()`，不让它直接反传梯度。

相对 V0 的变化：

- V0：只用真实 positive 和显式 negative。
- 这里：额外构造“介于正例和难负例之间”的合成难例，增加判别难度。

一句话总结：

```text
HardNegativeNLLLoss = V0 + MixCSE 风格的 row-wise mixed negative
```

### `HardNegativeNLLLossV1.py`

从实现上看，它与当前 `HardNegativeNLLLoss.py` 基本一致。

相对 V0 的核心增量也相同：

- 引入 hard negative mining。
- 引入固定 `lam` 的 mixed negative。
- mixed negative 按 query 单独追加一列。
- 仍使用标准 CE。

与 `HardNegativeNLLLoss.py` 的关系：

- 逻辑几乎相同，可以看作同一阶段的 MixCSE 版本。
- 主要可见差异是默认 `scale` 取值不同，`V1` 默认是 `20.0`。

一句话总结：

```text
V1 = V0 + 固定 lam 的 MixCSE mixed negative
```

### `HardNegativeNLLLossV2.py`

核心增量：

- 继承 V1 的 MixCSE mixed negative 设计。
- 在最终 CE 上加入 focal-style sample reweighting。
- 新增超参 `gamma`。

相对 V0 的变化：

- 不仅加入 mixed negative，还改变了样本加权方式。
- 对于正类概率低的难样本，赋予更高权重：
  `w_i = (1 - p_i) ^ gamma`

影响：

- 更强调难样本。
- `gamma=0` 时退化回普通 CE。

一句话总结：

```text
V2 = V1 + focal-style 难样本重加权
```

### `HardNegativeNLLLossV3.py`

核心增量：

- 不再使用固定 `lam`，而是使用 sample-wise dynamic `lam_i`。
- `lam_i` 由当前样本的难度决定：hard negative 越接近 positive，`lam_i` 越大。
- 在基础 CE 之外，再加入一个针对 mixed negative 的 margin-aware penalty。

相对 V0 的变化：

- V0 没有 mixed branch，也没有 margin 约束。
- V3 同时做了两层增强：
  1. mixed negative 从固定插值变成按难度自适应插值；
  2. 额外显式惩罚 mixed negative 过于接近 query 的情况。

影响：

- 比 V1/V2 更强调“合成难例必须真的难，但又不能太靠近正例”。
- loss 结构从单一 CE 变成 `base_loss + mix_weight * mix_loss`。

一句话总结：

```text
V3 = V1 的自适应 lam 版本 + mixed negative 的 margin penalty
```

### `HardNegativeNLLLossV4.py`

核心增量：

- 保留 V1 的“每行一个 mixed negative”框架。
- 将 mixed negative 的构造从普通线性插值扩展为：
  - `lerp`
  - `slerp`
- 默认更强调球面插值 `slerp`。

相对 V0 的变化：

- 除了引入 mixed negative，还把“怎么混合 pos 和 hard neg”显式参数化了。
- `slerp` 在归一化空间上插值，更贴近角度几何。

影响：

- 对 embedding 空间几何更敏感。
- 适合本项目里后续的 SlerpMixCSE 系列实验。

一句话总结：

```text
V4 = V1 的插值方式升级版，支持 SlerpMixCSE
```

### `HardNegativeNLLLossV5.py`

核心增量：

- 保留 V4 的 Slerp/Lerp mixed negative 机制。
- 但 mixed negative 不再严格“每行只用自己那一个”。
- 先为 batch 中每个样本构造一个 mixed negative，形成共享 mixed pool。
- 每个 query 再从这个共享 mixed pool 中选择 top-k 个最相似 mixed negatives。

相对 V0 的变化：

- V0 只有 `[positive | explicit negative]`。
- V5 变为 `[positive | explicit negative | top-k shared mixed negative]`。

影响：

- 让 mixed negative 从 row-wise 私有难例，扩展为 batch 内共享难例来源。
- 比 V4 产生更丰富、更强的困难候选。

一句话总结：

```text
V5 = V4 + top-k shared mixed negatives
```

### `HardNegativeNLLLossV6.py`

核心增量：

- retrieval 主体基本延续 V5。
- 额外新增 `aux_loss` / `aux_loss_weight` 接口。

相对 V0 的变化：

- 一方面继承了 V5 的 shared top-k SlerpMixCSE。
- 另一方面和 `V0StructuredSelfAttnAblation` 一样，支持外部辅助正则项。

最终形式：

```text
retrieval_loss + aux_loss_weight * aux_loss
```

一句话总结：

```text
V6 = V5 + 可选辅助损失
```

### `HardNegativeNLLLossV7AnglE.py`

核心增量：

- 在 V6 基础上，保留：
  - top-k shared mixed negatives
  - 可选 aux loss
  - Slerp/Lerp mixed interpolation
- 但相似度函数不再局限于 cosine。
- 新增 AnglE 风格 `angle_sim`，并支持：
  - 纯 angle
  - cosine + angle 混合打分

相对 V0 的变化：

- V0 只使用 cosine similarity。
- V7 直接替换/扩展了底层相似度定义。
- 因而它不仅是 loss 结构变化，也是“相似度空间”本身的变化。

影响：

- 更强调角度结构信息。
- 当 `cosine_weight > 0` 且 `angle_weight > 0` 时，相当于混合相似度训练。

一句话总结：

```text
V7AnglE = V6 + AnglE 风格相似度 / cosine-angle hybrid
```

## 快速对照表

| 文件 | 相对 V0 的主要增量 |
| --- | --- |
| `HardNegativeNLLLossV0_1.py` | 检测重复 negative，并将其从 softmax 中 mask 掉 |
| `HardNegativeNLLLossV0_2.py` | 显式 negative 改为与 query 按行对齐，只作用于本行 |
| `HardNegativeNLLLossV0_3.py` | 把重复 negative 提升为额外正例，改用 multi-positive NLL |
| `HardNegativeNLLLossV0StructuredSelfAttnAblation.py` | 在 V0 基础上增加可选辅助损失 |
| `HardNegativeNLLLoss.py` | 引入 MixCSE 风格的 hard negative mining 和 mixed negative |
| `HardNegativeNLLLossV1.py` | 固定 `lam` 的 MixCSE mixed negative 版本 |
| `HardNegativeNLLLossV2.py` | 在 V1 上加入 focal-style 难样本重加权 |
| `HardNegativeNLLLossV3.py` | 在 V1 上加入动态 `lam` 和 mixed-negative margin penalty |
| `HardNegativeNLLLossV4.py` | 在 V1 上将混合方式升级为 `lerp/slerp` |
| `HardNegativeNLLLossV5.py` | 在 V4 上将 mixed negative 扩展为 top-k shared mixed pool |
| `HardNegativeNLLLossV6.py` | 在 V5 上增加可选辅助损失 |
| `HardNegativeNLLLossV7AnglE.py` | 在 V6 上引入 AnglE 风格相似度或 cosine-angle 混合相似度 |

## V0_2 派生版本

当前旧版 loss 文件均保留为历史存档，不重命名、不移动、不改变默认注册入口。新增的 `*_2` 文件表示“保留原版本实验思想，但把 raw explicit negative 改为 V0_2 的 row-aligned 语义”。

共同规则：

- `d_reps_neg` 只能为空，或与 `q_reps` / `d_reps_pos` 行数一致。
- raw `neg_i` 只参与 `query_i` 自己那一行，不再作为全 batch 共享 negative pool。
- 若 `neg_i` 与 `pos_i` 完全相同，该行 raw negative 会被 mask。
- mixed negative 由 row-aligned `(pos_i, neg_i)` 生成；若该 raw negative 被判定为 duplicate positive，对应 mixed negative 也会从候选中 mask。

| 文件 | 相对旧版本的 V0_2 化策略 |
| --- | --- |
| `HardNegativeNLLLossV0_2StructuredSelfAttnAblation.py` | V0_2 retrieval loss + 可选 aux loss |
| `HardNegativeNLLLossV1_2.py` | 固定 `lam`，用本行 `neg_i` 生成本行 mixed negative |
| `HardNegativeNLLLossV2_2.py` | V1_2 + focal-style 难样本重加权 |
| `HardNegativeNLLLossV3_2.py` | 用 `s(q_i,pos_i)` 与 `s(q_i,neg_i)` 计算动态 `lam_i` 和 mixed margin penalty |
| `HardNegativeNLLLossV4_2.py` | V1_2 + `lerp/slerp` mixed 构造 |
| `HardNegativeNLLLossV5_2.py` | raw negative 不共享；只共享由 row-aligned pair 生成后的 mixed pool，并做 top-k |
| `HardNegativeNLLLossV6_2.py` | V5_2 + 可选 aux loss |
| `HardNegativeNLLLossV7_2AnglE.py` | V6_2 + AnglE / cosine-angle hybrid similarity |

## 演化主线总结

如果按演化路线看，大致可以分成三条线：

### 1. V0 系列修补线

- `V0_1`：处理最直接的 duplicate false negative。
- `V0_2`：处理“negative 不该跨样本共享”的问题。
- `V0_3`：把 duplicate false negative 从“删除”升级为“并入正例”。

这一条线的重点是：

```text
修补 V0 在负例定义上的监督偏差
```

### 2. MixCSE / SlerpMixCSE 主线

- `HardNegativeNLLLoss` / `V1`：引入 mixed negative。
- `V2`：加入 focal-style 难样本加权。
- `V3`：引入动态 lam 和 mixed margin penalty。
- `V4`：引入 slerp。
- `V5`：引入 shared top-k mixed pool。

这一条线的重点是：

```text
从“用真实负例训练”走向“主动构造更难的合成负例训练”
```

### 3. 辅助监督 / 相似度增强线

- `V0StructuredSelfAttnAblation`：V0 + aux loss。
- `V6`：V5 + aux loss。
- `V7AnglE`：V6 + angle similarity。

这一条线的重点是：

```text
在 retrieval loss 之外叠加结构先验或替换底层相似度
```

## 备注

- 本文档主要依据当前代码实现总结“实际行为”，不是根据文件名推测。
- `HardNegativeNLLLoss.py` 和 `HardNegativeNLLLossV1.py` 当前实现几乎一致，可以视为同一阶段的 MixCSE 版本。
- 若后面你还会继续加 `V8/V9/...`，最自然的写法是继续按“相对 V0 的增量 + 相对上一版的关键变化”追加在本文件里。
