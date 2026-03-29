# Ditto 对 DermL2V 的可迁移方法总结

## 背景

- 论文：Ditto: A Simple and Efficient Approach to Improve Sentence Embeddings
- 来源：https://arxiv.org/pdf/2305.10786.pdf
- 本文档目标：提炼 Ditto 中可迁移到 DermL2V 的部分，重点关注 pooling 设计、免训练重加权和方法论上的验证方式。

## 核心结论

Ditto 的核心不是“再加一个新的可训练模块”，而是“复用编码器内部已经存在的重要性信号”。
论文认为，vanilla mean pooling 会偏向不重要 token，因此使用某个选定 attention head 的对角自注意力值作为 token 权重。
这样就得到一种无需训练的加权 pooling 方法，在冻结编码器的情况下也能提升句向量质量。

对我们来说，可迁移的启发比 Ditto 的原始公式更广：

- token pooling 不应默认所有 token 贡献相等
- 可以从模型内部挖掘重要性信号，而不必额外增加参数
- 在上更大的 pooling 模块前，先验证轻量、尽量保留几何结构的重加权是否已经足够带来收益

## 可迁移创新 1：基于内部信号的 token 重加权

## Ditto 做了什么

- 对每个 token 提取某个 self-attention head 的对角项 `A_ii`。
- 用 `A_ii` 作为 token 重要性权重。
- 用加权 pooling 替代均匀平均 pooling。
- 论文中表现最好的设置之一，是对第一层与最后一层状态（`first-last`）做带权求和。

## 为什么这很适合迁移

在 DermL2V 里，我们已经有这些怀疑：

- 通用 token 会稀释领域关键术语
- instruction 或模板 token 可能会主导 pooled embedding
- 大型可训练 pooler 在优化初期可能会破坏预训练几何

因此，一个免训练或近乎免训练的 token 重加权层，正好对准这些问题。

## 可直接尝试的改造方向

- 基于 attention 对角项的加权 mean：
  从一个或若干选定 head 提取 token 权重，替代 plain mean pooling。
- instruction-aware 加权 pooling：
  将现有 instruction masking / skip 逻辑，与剩余内容 token 上的 attention 权重结合。
- layer-fusion + 重要性加权：
  不只在最后一层 token 上用权重，也可以作用到 `first-last`、多层融合输出或 latent 候选上。
- 多头共识加权：
  不只依赖单个 head，而是对多个更符合领域 salience 的 head 进行平均或排序聚合。
- 残差式加权：
  使用 `w_i = 1 + alpha * score_i` 或归一化软权重，使初始化尽量接近 mean pooling，而不是彻底替换它。

## 对我们来说更具体的实现方向

建议先走低风险顺序：

1. 增加一个新的 pooling mode，例如 `attn_diag_weighted_mean`。
2. 支持几种权重来源：
   - 一个固定的 `(layer, head)`
   - top-k heads 的平均
   - 只对非 instruction token 使用对角 attention
3. 在已有 token state 基础上比较：
   - `last`
   - `first_last`
   - 当前的 layer fusion 输出
4. 优先尝试残差式或归一化加权，减少几何漂移。

## 可迁移创新 2：把 head 选择当成一个小型搜索问题

Ditto 并不假设每个 attention head 都有用。
它会在开发集上对所有 head 做一次网格搜索，再选择最优的 `(layer, head)`。

这一点非常值得迁移。
与其一开始就设计一个大 pooler，不如先：

- 搜索哪些 head 与更好的检索或验证表现相关
- 看有用的 head 更集中在底层、中层，还是 instruction 较重的层
- 把 head search 当作诊断工具，再决定是否值得上更复杂的 pooling 结构

## 值得复用的操作方式

- 保持搜索空间小且显式
- 把 head 选择视为超参数，而不是定理
- 针对不同 backbone 和 pooling backbone 记录被选中的 head
- 测试 head 选择在不同 seed 和数据混合下是否稳定

## 可迁移创新 3：先做机制分析，再设计 pooling

Ditto 的流程是先提出机制性判断，再把判断转成方法：

- 判断 1：高质量句向量依赖于 informative token 的组合性
- 判断 2：某些 attention head 已经编码了 token 重要性
- 方法：把这些信号拿来做 pooling

这种“分析 -> 假设 -> 轻量方法”的工作流，在我们项目里也非常值得复用。

## DermL2V 中适合复用的模式

- 分析当前最好和最差 pooling 变体的 token 重要性行为
- 看强模型是否更关注医学上关键的 span
- 判断失败是否来自“所有 token 被同等 pooling”
- 然后再把分析信号提升成正式的 pooling 权重

这比直接跳到更大的可训练模块更稳。

## 可迁移佐证方法 1：验证好 embedding 是否更强调重要 token

论文用 perturbed masking 对比 BERT 和 SBERT，并展示出更强的句向量模型会更集中地依赖 informative token。

我们不一定需要一模一样复现 perturbed masking，但很适合复用它的验证逻辑：

- 比较一个较弱基线和一个较强基线
- 衡量更好的 embedding 是否更强调领域关键 token
- 用这个证据支撑为什么某个 reweighting pooler 是合理的

## 在 DermL2V 里的替代做法

- token ablation / leave-one-token-out 对最终 embedding 相似度的影响
- 观察 mask 掉哪些 token 会最明显改变检索排序
- 比较以下方法的影响集中度：
  - plain mean pooling
  - latent pooling
  - residual MLP pooling
  - attention-weighted pooling
- 看疾病名、形态学术语、解剖部位、严重程度描述、否定词等是否获得更高贡献

## 可迁移佐证方法 2：把模型内部 token 分数和外部显著性信号做相关性分析

Ditto 用以下关系来验证其 weighting signal：

- impact score 与 TF-IDF 的相关性
- 对角 attention 与 TF-IDF 的相关性

这个模式本身非常重要，即使 TF-IDF 不一定是我们最理想的代理指标。

## 适合我们领域的外部或伪外部信号

- 皮肤科语料上的 corpus TF-IDF
- 疾病术语和皮损术语的 ontology / vocabulary 命中标记
- 人工整理的显著 token 词表
- 如果数据里有 rationale、finding 或 entity span，也可以直接利用 span 标签
- 来自更强 teacher 的 retrieval-gradient 或 cross-encoder attribution 分数

## 推荐检查

在信任任何新权重信号之前，至少验证它和一个外部显著性代理存在相关性。
如果相关性很弱，那么性能提升可能只是偶然或者不可泛化。

## 可迁移佐证方法 3：同时验证任务指标与嵌入几何

Ditto 不只报告 STS 准确率。
它还检查了：

- alignment / uniformity
- 平均 cosine similarity，作为 isotropy 的代理
- 被选中 head 与 TF-IDF 的相关性

这对我们的项目尤其重要，因为有些 pooler 可能会让训练 loss 变好，但同时悄悄破坏 embedding geometry。

## 我们可以复用的指标

- 下游检索 / 匹配指标
- 验证集相关性或排序指标
- alignment / uniformity
- cosine similarity 分布漂移
- 随机样本之间的平均非对角 cosine
- 主成分解释方差

## 为什么这很重要

一个新的 pooling 设计可能只在某一个指标上看起来更好，但实际上已经让表示空间开始塌缩。
Ditto 的证据风格之所以有价值，是因为它明确拆开了：

- 任务收益
- 显著性对齐
- 表示几何

这种拆分会让我们的 retrospective 更扎实。

## 可迁移佐证方法 4：和更强的 oracle-like weighting 做对照

论文把 Ditto 与 TF-IDF 加权 pooling 做了比较。
这是一个非常好的验证模式：

- 如果一个便宜的内部信号能够逼近更强的外部加权基线，方法就更可信
- 如果差距很大，说明这个内部信号大概率只捕捉到了部分真实显著性结构

## DermL2V 版本的这个测试

建议比较：

- mean pooling
- 基于 attention 对角项的加权 pooling
- TF-IDF 加权 pooling
- ontology 加权 pooling
- teacher attribution 加权 pooling

解释方式：

- 如果 attention weighting 接近 teacher 或 ontology weighting，它就是一个很强的轻量方法
- 如果只有外部 weighting 有效，说明当前编码器内部可能还没有形成可用的显著性表示

## 可复用的实验模板

## 阶段 1：免训练筛选

- 冻结 encoder。
- 评估 `mean`、`first_last` 和 `attn_diag_weighted_mean`。
- 在小验证子集上做 head search，而不是直接全量训练。
- 看收益是否在不同 seed 和数据集上稳定。

## 阶段 2：几何与显著性验证

- 测量 cosine similarity 的塌缩或扩散
- 测量 alignment / uniformity
- 计算 token 权重与皮肤科显著性代理之间的相关性
- 对少量临床关键样本做 token 级 case study

## 阶段 3：受控整合

- 把 weighting 和当前最佳 pooling backbone 结合
- 在上可训练 pooler 之前，先尝试残差式 weighting
- 只有在轻量 weighting 确实触顶后，才升级到 latent 或 MLP pooler

## 值得新增的具体变体

- `attn_diag_weighted_mean`
- `attn_diag_weighted_first_last`
- `attn_diag_topk_mean`
- `attn_diag_residual_mean`
- `attn_diag_instruction_masked_mean`

## 风险与注意事项

- Ditto 的 head 选择依赖开发集搜索，因此收益可能部分来自搜索运气。
- 论文只在英文 STS 上评估，迁移到皮肤科检索并没有保证。
- 对角 attention 可能追踪的是词频或格式伪迹，而不是真正的临床显著性。
- 单 head 的加权规则可能在不同 backbone 上不稳定。
- 如果我们的模型在监督下已经能抑制无信息 token，那么额外 weighting 的收益可能很小。

## 结论

Ditto 最值得借鉴的，主要是它的设计哲学和验证模板：

- 先挖掘内部重要性信号
- 再构造最小化扰动的重加权 pooler
- 然后用显著性相关性和几何诊断去验证

对 DermL2V 来说，最高价值的迁移不一定是论文里的原始公式本身，而是一整组基于 attention 的、低破坏性的加权 pooling 基线。它们应该先于更大的可训练 pooling 模块被测试。

## 立即可做的下一步检查

- [ ] 加一个基于 attention 对角权重的轻量 pooling 原型。
- [ ] 先在 `mean` 和 `first_last` 上评估，再考虑更大的 pooler。
- [ ] 做一个显著性代理相关性分析脚本或 notebook。
- [ ] 在未来的 pooling 实验里，把几何诊断和检索指标一起记录。
