## 2026-03-22

### Session Summary

本次 session 主要围绕 NeurIPS 风格的 introduction 组织方式，以及 `Our Contributions` 部分的叙事层级展开。首先明确了引言整体应保持方法论文定位，而不是应用论文定位；虽然结构上继续采用 `Background`、`Phenomenon & Challenge`、`Our Contributions` 三段式，但写作上要围绕一条更强的论证主线展开：皮肤病文本建模的核心难点不是一般意义上的医学领域适配，而是高噪声、非结构化、跨来源异构文本中的临床语义异构性，其本质可表述为一个兼顾 invariance 与 discrimination 的 clinical semantic representation learning 问题。

在 `Our Contributions` 的组织方式上，形成了一个明确共识：NeurIPS 风格下更有说服力的写法，不应是简单列举模块或技术名词，而应采用 `challenge -> method response` 的映射方式来组织贡献。当前比较合理的三层结构是：

1. 针对现有医学文本编码器在 noisy and heterogeneous dermatology text 上 sentence-level clinical alignment 能力不足的问题，构建一个以 `LLM-based encoder + supervised contrastive learning` 为核心的表示学习框架。
2. 针对模型难以对跨来源、跨风格、跨噪声条件下临床等价表达保持不变性的问题，设计 challenge-aware augmentation，显式放大 formally heterogeneous but clinically equivalent views，强化 invariance learning 的监督信号。
3. 针对皮肤病文本中细粒度术语、部位和程度差异容易被弱化的问题，设计 `TopKshareSlerpMixCSE`，增强 diagnostically meaningful fine-grained discrimination。

围绕第一点贡献的表述方式，进行了重点讨论。当前实际方法中，替换为 LLM-based encoder 后，曾对比三种训练方式：`token-level MNTP`、基于两次 dropout 前向构造正样本的 `unsupervised contrastive learning`、以及 `supervised contrastive learning`。最终选择 supervised contrastive learning 作为训练目标。这里的关键写作问题在于，若直接表述为“换成更强 encoder 并选择更优训练方式”，容易被审稿人理解为 engineering upgrade，而不是方法贡献。

对此形成的结论是：第一点不能写成简单的 backbone replacement，也不宜夸大成“提出了一个全新的 loss”。更稳妥且更具 NeurIPS 风格的写法，是将其上升为一个 `objective mismatch` 问题：现有 token-level language modeling 或 instance-level unsupervised contrastive objectives，与 dermatology text 所要求的 sentence-level clinical alignment 之间存在错配。因此，第一点贡献应被表述为“重新审视皮肤病文本表征学习的优化目标，并通过系统比较不同训练范式，确定 supervised sentence-level contrastive optimization 是更契合该问题结构的训练原则”，再在此基础上构建 `LLM-based encoder + supervised contrastive learning` 的表示学习框架。

进一步讨论后，明确了本文当前可以被组织成一个“三层递进设计”而非“三个并列模块”：基础层是训练原则与表示学习框架的选择；第二层是不变性建模，即通过数据增强强化 clinically equivalent alignment；第三层是判别性建模，即通过 `TopKshareSlerpMixCSE` 保留 fine-grained diagnostic distinctions。这种写法可以在不强行扩展模块数量的前提下，增强整篇方法论文的完整性和说服力。

最后达成的阶段性判断是：现阶段无需急于将贡献扩展为四点。当前更重要的是先把三点贡献的层级关系和论证方式写扎实，尤其是将第一点稳定地写成“训练目标与表示学习框架层面的贡献”，而不是背景叙述或工程替换。后续如果需要增强支撑力度，再考虑是否将“问题重构”单独抽出作为一条独立 contribution。
