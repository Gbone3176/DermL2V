# Introduction

## Background

### 中文

医学语言表征模型（如 PubMedBERT、ClinicalBERT 等）通过将领域知识嵌入参数空间，获得了将医学文本映射为密集向量的强大表征能力。正是基于这种语义表示基础，它们不仅被广泛应用于医学检索、分类和语义匹配等纯文本任务，同时也在医学多模态任务中充当了至关重要的桥梁。传统的表征模型主要依赖于掩码语言建模等预训练范式，其核心优势在于局部上下文的捕获与 token 级别的医学术语匹配。

近年来，基于大语言模型（LLMs）的表征模型凭借其庞大的参数空间, 上下文长度和海量的预训练知识，进一步推动了自然语言表征领域的范式转移。这些模型显著提升了长文本理解和复杂上下文语义建模的能力，为处理更高复杂度的文本任务提供了新的可能。

### English

Medical language representation models, such as PubMedBERT and ClinicalBERT, embed domain knowledge into their parameter spaces to acquire the powerful representation capability of mapping medical text into dense vectors. Building upon this semantic foundation, they have been widely adopted not only for pure-text tasks like biomedical retrieval, classification, and semantic matching, but also as a crucial bridge in medical multimodal applications. Traditionally, these models rely on pre-training paradigms like masked language modeling, which excel at capturing local context and token-level medical terminology matching. 

Recently, representation models based on large language models (LLMs) have driven a paradigm shift in natural language representation. Leveraging massive parameter spaces and extensive pre-trained knowledge, these LLM-based models significantly enhance long-context understanding and complex semantic modeling, opening new possibilities for processing highly complex text tasks.

## Phenomenon & Challenge

### 中文

尽管医学表征模型进展显著，但直接应用于高噪声临床领域（如皮肤病学）时仍面临根本性鸿沟。真实临床文本高度非结构化且来源异构，其核心瓶颈在于严重的领域内临床语义异构性。局部词汇匹配与通用上下文理解均无法跨越此变异以实现“临床语义对齐”。具体而言，该领域表征学习面临三重挑战：一是传统医学文本表征模型的局限性。受限于较短的上下文窗口与有限的参数空间，传统模型不仅在处理文本时常因被迫裁切而丢失关键信息，也难以充分表征复杂的医学语义；同时，针对不同的下游任务，传统的医学编码器通常只能提供相同的表征，缺乏任务感知能力。二是皮肤病文本表层变异下的临床等价性。等价病例常表达迥异（如患者的“胳膊痒掉皮”与医生的“红斑伴鳞屑”毫无词汇交集但语义相近），要求模型具备跨越风格噪音的语义不变性；三是细粒度文本差异下的语义分歧。仅将“鳞屑”替换为“水疱”，文本形式几乎未变，诊断含义却完全反转。由于训练目标错配，现有模型难以兼顾“等价对齐”与“诊断敏感性”，无法满足高异构场景下句子级临床语义对齐的需求。

### English

Despite significant progress, applying representation models directly to noisy clinical domains (e.g., dermatology) exposes a fundamental gap. Real-world clinical text is highly unstructured and heterogeneous. The core bottleneck is severe intra-domain semantic heterogeneity. Neither local vocabulary matching nor general contextual understanding can bridge these surface variations to achieve precise "clinical semantic alignment." Specifically, representation learning faces a triple challenge: First, *the limitations of traditional medical text representation models*. Constrained by shorter context windows and limited parameter spaces, traditional models not only frequently lose critical semantics due to forced truncation when processing texts, but also struggle to adequately represent complex medical concepts. Meanwhile, for different downstream tasks, traditional medical encoders typically provide identical representations, lacking task-aware capabilities. Second, *clinical equivalence under surface variation*. Equivalent cases are expressed drastically differently (e.g., a patient's "itchy elbows with flaking" vs. a clinician's "erythematous plaques with scale" share no lexical overlap but identical meanings), requiring semantic invariance across stylistic noise. Third, *semantic divergence under subtle textual differences*. Simply replacing "scale" with "vesicles" leaves the surface form nearly unchanged but completely flips the diagnosis. Due to mismatched training objectives, existing models fail to simultaneously filter stylistic noise and capture subtle diagnostic differences, falling short of sentence-level clinical alignment requirements.

## Our Contributions    

### 中文

应对上述挑战的核心在于解决标准目标与“句子级临床语义对齐”的系统性错配。为此，本文提出一种针对复杂医学文本的表示学习框架：首先，引入**指令遵循的有监督对比学习框架**，利用 LLM 的长输入上下文和庞大参数空间，将训练约束条件与句级临床信息表征目标对齐，同时引入指令遵循方法赋予模型任务感知能力；其次，针对“临床等价性”挑战，设计**临床等价性感知的数据增强策略**，构造形式异构但语义一致的视图以强制模型学习语义不变性；最后，针对“语义分歧”脆弱性，提出 **S-MixCSE**，通过超球面困难负样本混合保留模型对细微差异的诊断敏感性。实验表明，该框架在多项皮肤病文本任务上取得稳定的 SOTA 性能，为异构医学文本建模提供了可迁移的方法论。

### English

Addressing these challenges fundamentally relies on resolving the systematic mismatch between standard learning objectives and "sentence-level clinical semantic alignment." To this end, we propose a representation learning framework tailored for complex medical texts. First, we introduce an **instruction-following supervised contrastive learning framework**. By leveraging the long input context and massive parameter space of LLMs, this framework aligns training constraints with sentence-level clinical representation objectives, while incorporating an instruction-following mechanism to endow the model with task-aware capabilities. Second, to tackle the "clinical equivalence" challenge, we design a **clinical equivalence-aware data augmentation strategy** that explicitly constructs formally heterogeneous yet semantically consistent views to enforce semantic invariance. Finally, to address the vulnerability of "semantic divergence," we propose **S-MixCSE**, which leverages hyperspherical hard negative mixing to preserve the model's diagnostic sensitivity to fine-grained differences. Extensive experiments demonstrate that our framework achieves stable state-of-the-art (SOTA) performance across multiple dermatology text tasks, providing a transferable methodology for heterogeneous medical text modeling.


## Section Summary

### 中文

总结而言，本文的核心工作可凝练为三点：（1）揭示了现有表示学习目标与高异构医学文本所需的“句子级临床语义对齐”之间存在的系统性错配问题；（2）提出了一套以指令遵循的监督对比优化为核心，融合临床等价性感知数据增强与 S-MixCSE 困难负样本合成策略的原则性表示学习框架；（3）在大量复杂的下游实验任务中全面验证了该方法的有效性，确立了其在提升临床表示鲁棒性与诊断敏感性方面的核心价值。

### English

In summary, our core work can be distilled into three aspects: (1) We identify the systematic mismatch between standard representation learning objectives and the sentence-level clinical alignment required by highly heterogeneous medical text; (2) We propose a principled representation learning framework centered on instruction-following supervised contrastive optimization, integrated with clinical equivalence-aware data augmentation and S-MixCSE hard negative synthesis; and (3) We comprehensively validate the effectiveness of our approach across extensive downstream experiments, establishing its substantial value in enhancing both the robustness and diagnostic sensitivity of clinical representations.


## Discussion Notes

> **Review Check (Addressed)**:
> - `Our Contributions` structurally refactored into a clear `Challenge -> Method Response` narrative.
> - Explicit emphasis was placed on the `objective mismatch` across MNTP/Unsup/Sup paradigms to avoid the "backbone replacement" perception.
> - Added explicit headers for the contributions matching the theoretical challenges.

## Writing Notes

> **Review Check (Addressed)**:
> - Condensed medical model history and firmly pivoted to the methodology-driven paradigm shift.
> - Emboldened and centralized the dual challenge: `clinical equivalence under surface variation` vs `semantic divergence under subtle textual differences`.
> - Contributions are stated at a high, principled level to easily align with subsequent Methods/Experiments sections.


# Latex Version

```latex
\section{Introduction}

Medical language representation models, such as PubMedBERT and ClinicalBERT, embed domain knowledge into their parameter spaces to acquire the powerful representation capability of mapping medical text into dense vectors. Building upon this semantic foundation, they have been widely adopted not only for pure-text tasks like biomedical retrieval, classification, and semantic matching, but also as a crucial bridge in medical multimodal applications. Traditionally, these models rely on pre-training paradigms like masked language modeling, which excel at capturing local context and token-level medical terminology matching. Recently, representation models based on large language models (LLMs) have driven a paradigm shift in natural language representation. Leveraging massive parameter spaces and extensive pre-trained knowledge, these LLM-based models significantly enhance long-context understanding and complex semantic modeling, opening new possibilities for processing highly complex text tasks.

Despite significant progress, applying representation models directly to noisy clinical domains (e.g., dermatology) exposes a fundamental gap. Real-world clinical text is highly unstructured and heterogeneous. The core bottleneck is severe intra-domain semantic heterogeneity. Neither local vocabulary matching nor general contextual understanding can bridge these surface variations to achieve precise ``clinical semantic alignment.'' Specifically, representation learning faces a triple challenge: First, \textit{the limitations of traditional medical text representation models}. Constrained by shorter context windows and limited parameter spaces, traditional models not only frequently lose critical semantics due to forced truncation when processing texts, but also struggle to adequately represent complex medical concepts. Meanwhile, for different downstream tasks, traditional medical encoders typically provide identical representations, lacking task-aware capabilities. Second, \textit{clinical equivalence under surface variation}. Equivalent cases are expressed drastically differently (e.g., a patient's ``itchy elbows with flaking'' vs. a clinician's ``erythematous plaques with scale'' share no lexical overlap but identical meanings), requiring semantic invariance across stylistic noise. Third, \textit{semantic divergence under subtle textual differences}. Simply replacing ``scale'' with ``vesicles'' leaves the surface form nearly unchanged but completely flips the diagnosis. Due to mismatched training objectives, existing models fail to simultaneously filter stylistic noise and capture subtle diagnostic differences, falling short of sentence-level clinical alignment requirements.

Addressing these challenges fundamentally relies on resolving the systematic mismatch between standard learning objectives and ``sentence-level clinical semantic alignment.'' To this end, we propose a representation learning framework tailored for complex medical texts. First, we introduce an \textit{instruction-following supervised contrastive learning framework}. By leveraging the long input context and massive parameter space of LLMs, this framework aligns training constraints with sentence-level clinical representation objectives, while incorporating an instruction-following mechanism to endow the model with task-aware capabilities. Second, to tackle the ``clinical equivalence'' challenge, we design a \textit{clinical equivalence-aware data augmentation strategy} that explicitly constructs formally heterogeneous yet semantically consistent views to enforce semantic invariance. Finally, to address the vulnerability of ``semantic divergence,'' we propose \textit{S-MixCSE}, which leverages hyperspherical hard negative mixing to preserve the model's diagnostic sensitivity to fine-grained differences. Extensive experiments demonstrate that our framework achieves stable state-of-the-art (SOTA) performance across multiple dermatology text tasks, providing a transferable methodology for heterogeneous medical text modeling.

In this paper, our core work can be distilled into three aspects: (1) We identify the systematic mismatch between standard representation learning objectives and the sentence-level clinical alignment required by highly heterogeneous medical text; (2) We propose a principled representation learning framework centered on instruction-following supervised contrastive optimization, integrated with clinical equivalence-aware data augmentation and S-MixCSE hard negative synthesis; and (3) We comprehensively validate the effectiveness of our approach across extensive downstream experiments, establishing its substantial value in enhancing both the robustness and diagnostic sensitivity of clinical representations.
```