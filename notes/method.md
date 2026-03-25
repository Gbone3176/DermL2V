# Method

## Spherical Mixup with Hard Negative Contrastive Learning

我们采用一种基于显式难负样本挖掘的对比学习框架，以学习更具判别性的文本表示。对于一个训练 batch，设共有 $B$ 个查询-正样本对，记第 $i$ 个查询为 $q_i$，其对应正样本为 $p_i$，并额外给定一个显式负样本池 $\mathcal{N}=\{n_j\}_{j=1}^{N}$。编码器 $f(\cdot)$ 将输入文本映射到 $d$ 维向量空间，从而得到查询表示 $\mathbf{h}^q_i=f(q_i)$、正样本表示 $\mathbf{h}^p_i=f(p_i)$，以及显式负样本表示 $\mathbf{h}^n_j=f(n_j)$。训练目标是在 batch 内检索正确的配对正样本，同时利用显式负样本和混合负样本提高判别边界的鲁棒性。

我们首先构建标准的 batch 内对比候选集。对任意查询表示 $\mathbf{h}^q_i$ 与正样本表示 $\mathbf{h}^p_k$，采用余弦相似度

$$
\mathrm{sim}(\mathbf{x},\mathbf{y})=\left\langle \frac{\mathbf{x}}{\|\mathbf{x}\|_2}, \frac{\mathbf{y}}{\|\mathbf{y}\|_2} \right\rangle,
$$

并通过温度缩放系数 $s$ 得到 batch 内正样本候选 logits：

$$
L^{\mathrm{pos}}_{ik}= s \cdot \mathrm{sim}(\mathbf{h}^q_i,\mathbf{h}^p_k), \quad 1 \le i,k \le B.
$$

其中，对角元素 $L^{\mathrm{pos}}_{ii}$ 对应真实正样本，其余 batch 内样本自然充当 in-batch negatives。与此同时，我们进一步利用显式负样本池计算

$$
L^{\mathrm{neg}}_{ij}= s \cdot \mathrm{sim}(\mathbf{h}^q_i,\mathbf{h}^n_j), \quad 1 \le j \le N.
$$

对于每个查询 $q_i$，我们从显式负样本池中选择与其最相似的样本作为 hardest negative：

$$
j_i^{*} = \arg\max_{1 \le j \le N} \mathrm{sim}(\mathbf{h}^q_i,\mathbf{h}^n_j), \qquad
\mathbf{h}^{hn}_i = \mathbf{h}^n_{j_i^{*}}.
$$

该步骤使模型始终关注当前最容易与正样本混淆的负例，而不是平均地对待所有负样本。与传统仅使用 hardest negative 的方法不同，我们进一步在表示空间中构造一个更具挑战性的混合负样本。具体而言，我们将配对正样本 $\mathbf{h}^p_i$ 与 hardest negative $\mathbf{h}^{hn}_i$ 进行插值，得到 query-specific mixed negative。在线性插值版本中，混合表示定义为

$$
\tilde{\mathbf{h}}_i = \lambda \mathbf{h}^p_i + (1-\lambda)\mathbf{h}^{hn}_i,
$$

其中 $\lambda \in [0,1]$ 为固定混合系数。为了更好地保持表示位于归一化嵌入流形上，我们在实际实现中采用球面线性插值（spherical linear interpolation, SLERP）。记归一化后的向量为

$$
\bar{\mathbf{h}}^p_i = \frac{\mathbf{h}^p_i}{\|\mathbf{h}^p_i\|_2}, \qquad
\bar{\mathbf{h}}^{hn}_i = \frac{\mathbf{h}^{hn}_i}{\|\mathbf{h}^{hn}_i\|_2},
$$

二者夹角为

$$
\theta_i = \arccos \left( \left\langle \bar{\mathbf{h}}^p_i, \bar{\mathbf{h}}^{hn}_i \right\rangle \right).
$$

令插值参数 $t = 1-\lambda$，则 SLERP 混合表示写为

$$
\tilde{\mathbf{h}}_i =
\frac{\sin\left((1-t)\theta_i\right)}{\sin(\theta_i)} \bar{\mathbf{h}}^p_i +
\frac{\sin\left(t\theta_i\right)}{\sin(\theta_i)} \bar{\mathbf{h}}^{hn}_i.
$$

当 $\theta_i$ 非常小时，$\sin(\theta_i)$ 会导致数值不稳定，因此实现中退化为归一化后的线性插值作为 fallback。进一步地，我们对最终混合向量再次进行 $L_2$ 归一化。由于当前设置采用固定 $\lambda=0.2$，故 $t=0.8$，即混合向量整体更靠近 hardest negative，同时保留少量正样本语义，从而在语义边界附近构造出更难区分的负例。

在得到混合负样本后，我们仅为每个查询追加一个与自身对应的 mixed negative logit，而不是将所有混合样本共享为全局候选池。于是第 $i$ 个查询的 mixed-negative logit 可写为

$$
L^{\mathrm{mix}}_i = s \cdot \mathrm{sim}(\mathbf{h}^q_i,\tilde{\mathbf{h}}_i).
$$

最终，第 $i$ 行的候选集合由三部分组成：全部 batch 内正样本 logits、全部显式负样本 logits，以及该查询专属的 mixed negative logit，即

$$
\mathbf{z}_i = \left[
L^{\mathrm{pos}}_{i1}, \dots, L^{\mathrm{pos}}_{iB},
L^{\mathrm{neg}}_{i1}, \dots, L^{\mathrm{neg}}_{iN},
L^{\mathrm{mix}}_i
\right].
$$

由于真实目标始终是检索到与查询对齐的正样本 $p_i$，故监督标签保持为 batch 内对角位置 $y_i=i$。训练损失定义为标准交叉熵：

$$
\mathcal{L} = - \frac{1}{B} \sum_{i=1}^{B}
\log
\frac{\exp\left(L^{\mathrm{pos}}_{ii}\right)}
{\sum_{k=1}^{B} \exp\left(L^{\mathrm{pos}}_{ik}\right)
+ \sum_{j=1}^{N} \exp\left(L^{\mathrm{neg}}_{ij}\right)
+ \exp\left(L^{\mathrm{mix}}_i\right)}.
$$

该目标的关键在于，mixed negative 并不改变正样本标签，而是作为一个额外的高难度干扰项迫使模型学习更紧致的类内聚合和更清晰的类间分离。值得注意的是，在实现中混合分支采用 stop-gradient 策略，即 $\tilde{\mathbf{h}}_i$ 在构造后被视为常量，不通过该分支直接向正样本或 hardest negative 反向传播梯度。这样做的目的在于将 mixed negative 作为“判别压力”的来源，而非额外的表示学习目标，从而稳定训练过程并避免插值分支主导优化方向。

在分布式训练场景下，我们首先在所有设备之间收集查询、正样本和显式负样本表示，再统一执行 hardest negative 挖掘与 mixed negative 构造。因此，难负样本搜索和对比学习目标均建立在全局 batch 上，而非仅限于单卡局部 batch，这进一步提升了候选空间的多样性和训练难度。总体而言，该方法通过“batch 内对比 + 显式难负样本挖掘 + 球面 mixup 负样本构造”的联合设计，在不改变主任务标签定义的前提下，为检索式表示学习引入了更加细粒度且边界感知的训练信号。
