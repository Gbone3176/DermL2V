# INSIGHTS

## 2026-03-13 06:35:58 UTC

### 主题
HardNegativeNLLLossV1/V2 的算法观察、失败原因分析与可行创新方向。

### 代码观察
- `HardNegativeNLLLossV1.py` 中的 `lam` 是固定常数，混合方式为 `mixed = lam * pos + (1 - lam) * hard_neg`，且 `mixed.detach()`，因此它只作为额外负样本施压，不参与自身生成路径的梯度更新。
- `HardNegativeNLLLossV1.py` 中每个 query 只追加一个 `logits_mix`，不是把全部 mixed negatives 放入共享候选池，因此 mixed 分支对整体判别边界的影响比较弱，更像对每行增加一个“定制化困难负样本”。
- `HardNegativeNLLLossV2.py` 只是在最终 CE 上增加样本级 focal 权重 `w_i = (1 - p_i)^gamma`，没有改变 hard negative 的构造方式，也没有改变 mixed negative 的难度分布。

### 为什么固定 lam 值得创新
- 固定 `lam` 默认假设所有 query 的正负混合难度应该一致，但真实训练中不同 query 的正样本裕量、负样本混淆度、语义多样性都不同，这个假设通常过强。
- 当 `lam` 太小，mixed negative 更接近 hard negative，容易变成“重复强调已有 hardest negative”，新增信息有限。
- 当 `lam` 太大，mixed negative 更接近 positive，可能过度制造近邻伪负样本，导致表示空间被不必要地拉裂，尤其对语义边界本来就模糊的数据更危险。
- 因为当前 `mixed` 被 `detach`，固定 `lam` 一旦选得不好，模型只能被动适应这个噪声难度，无法自我修正。

### 可行创新 1：基于难度的动态 lam
- 核心想法：让 `lam_i` 随每个样本的难度动态变化，而不是全局固定。
- 一个直接定义：
  `hardness_i = sigmoid(alpha * (s(q_i, hard_i) - s(q_i, pos_i)))`
  `lam_i = lam_min + (lam_max - lam_min) * hardness_i`
- 解释：
  当 hardest negative 已经非常接近 positive 时，说明该 query 本身困难，应提高 `lam_i`，让 mixed negative 更靠近 positive，制造更细粒度决策边界。
  当 hardest negative 还很远时，降低 `lam_i`，避免过早制造伪负样本。
- 这个方案的优点是无需额外网络参数，训练稳定性通常也比可学习 MLP gate 更好。

### 可行创新 2：课程式 lam 调度
- 让 `lam` 随训练阶段变化，而不是对所有 step 恒定。
- 建议：
  前期小 `lam`，先学粗粒度分离；
  中后期逐步增大 `lam`，再逼近 fine-grained boundary。
- 一个简单形式：
  `lam_t = lam_start + (lam_end - lam_start) * progress^beta`
- 适用场景：
  数据规模较大、训练早期不稳定、hard negative 质量随 encoder 提升而改善时，这比从第一个 step 就用高难 mixed negative 更合理。

### 可行创新 3：分布式 lam 采样而不是单点 lam
- 不只构造一个 mixed negative，而是从区间 `[lam_low, lam_high]` 采样 2 到 4 个 `lam`，生成多个不同难度的 mixed negatives。
- 这等价于把单点边界扩展为一段“局部对抗带”，更接近 margin band regularization。
- 风险是 logits 数量增加，可能需要限制每个 query 的 mixed 个数，或者只保留 top-k 最困难的 mixed logits。

### 为什么 V2 的 focal 思想可能没有明显提升
- 问题 1：它是样本级重权，而不是负样本级重权。
  当前 focal 只根据正类概率 `p_i` 重加权整行 loss，无法区分“这行里到底是 in-batch negative 更难，还是 mixed negative 更难”。因此它没有直接改善 hardest confuser 的梯度分配。
- 问题 2：困难样本里混有噪声样本。
  对比学习中的低 `p_i` 不一定代表有价值的 hard case，也可能代表标注噪声、语义重叠、正样本质量差。focal 会把这些样本统一放大，容易伤害最终 embedding 几何结构。
- 问题 3：InfoNCE/CE 本身已经偏向 hardest negatives。
  softmax 的指数机制天然对高相似负样本更敏感，再叠加 focal 容易过度聚焦极难样本，造成优化目标过尖锐，损害全局均匀性。
- 问题 4：当前 mixed negative 只有一列。
  在 `B + N + 1` 个候选中，focal 放大的仍是整行 CE，而不是专门提升那一个 mixed logit 的作用，因此“mix 分支弱、focal 外层强”的组合可能并不匹配。
- 问题 5：若使用较大的 `scale`，softmax 会更尖锐。
  此时 `p_i` 很容易快速接近 0 或 1，focal 权重分布可能在训练早期过大、后期过小，导致收益不稳定。

### 两者如何融合得更合理
- 方向不是“动态 lam + 旧式 focal 直接叠加”，而是让 reweighting 感知 mixed negative 的难度来源。
- 推荐方案：Difficulty-Adaptive Mix + Margin-Aware Focal。

### 融合方案草图
- 第一步：计算三种分数
  `s_pos = s(q_i, pos_i)`
  `s_hard = s(q_i, hard_i)`
  `s_mix = s(q_i, mixed_i)`
- 第二步：定义 margin
  `m_hard = s_pos - s_hard`
  `m_mix = s_pos - s_mix`
- 第三步：动态确定 `lam_i`
  让 `lam_i` 由 `m_hard` 决定，而不是固定值。
  例如：
  `lam_i = lam_min + (lam_max - lam_min) * exp(-tau * relu(m_hard))`
- 第四步：不要对整行 CE 做 focal，而是只对 mixed branch 做 margin-aware weighting。
  例如定义：
  `w_mix_i = (1 + relu(m_target - m_mix))^gamma`
- 最终损失可以写成：
  `L = CE(logits_base) + lambda_mix * mean(w_mix_i * softplus(scale * (s_mix - s_pos + delta)))`
- 含义：
  基础 CE 维持整体检索判别；
  mixed branch 单独承担“逼近决策边界”的责任；
  reweighting 只在 `m_mix` 太小的时候增强，避免把整行 noisy sample 一起放大。

### 这个融合方案为什么比 V2 更可能有效
- 它把“难度建模”放到了 mixed negative 生成和 mixed branch 惩罚上，而不是粗糙地放到整行样本权重上。
- 它关注的是 margin，而不是单纯的正类概率。对对比学习来说，margin 往往比 `p_i` 更贴近我们真正关心的几何分离程度。
- 它能把“该不该加强这个样本”与“加强哪一种负样本压力”区分开，这是 V2 没做到的。

### 建议优先验证的 3 个实验
- 实验 A：固定 lam vs 课程式 lam。
  控制变量只改 `lam` 调度，先验证“固定常数是否是主要瓶颈”。
- 实验 B：固定 lam vs 样本级动态 lam。
  用 `m_hard` 或 `s_hard - s_pos` 驱动 `lam_i`，观察 STS/检索指标与训练稳定性。
- 实验 C：旧 V2 focal vs mixed-branch margin loss。
  保持 V1 主干不变，只把 V2 的样本级 focal 替换为 mixed branch 专属的 margin-aware penalty，验证 focal 失败是否源于施力位置错误。

### 实现优先级建议
- 第一优先级：样本级动态 `lam_i`，先不用额外可学习参数。
- 第二优先级：课程式 `lam_t` 与动态 `lam_i` 组合。
- 第三优先级：移除整行 focal，改成 mixed branch 的 margin-aware weighting。
- 暂不建议第一版就引入 MLP 预测 `lam_i`，因为可解释性差，且更容易把 gain 混入额外参数量而不是机制本身。
