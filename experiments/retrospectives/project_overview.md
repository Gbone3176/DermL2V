# 项目总览：Dermatology Domain Fine-tuning on LLM2Vec

## 1. 项目目标

这个项目的核心目标是：

- 以一个已经在大规模自然语言文本上预训练好的 LLM-based 文本编码器为基础
- 保留 LLM2Vec 将 decoder-only LLM 改造成文本编码器的思路
- 在皮肤病专科文本数据上继续进行领域微调
- 主要训练方式使用 LoRA，尽量用较低训练成本获得更适合 dermatology 语义检索/匹配的表示

一句话概括：

> 这是一个把通用 LLM 文本表征能力迁移到皮肤病专科文本上的 embedding / retrieval 项目。

## 2. 项目当前主线

当前主线不是从零训练一个新模型，而是在原始 LLM2Vec 框架上做领域适配，主要包括三层工作：

1. 保留 LLM2Vec 的 encoder 化能力
   - 支持 bidirectional attention
   - 支持 instruction-aware 输入格式
   - 支持多种 pooling 策略

2. 在皮肤病文本上做 LoRA 微调
   - 使用 dermatology 相关语料和构造好的 query-positive-negative 数据
   - 重点优化检索式/对比式 embedding 质量

3. 在模型结构和损失函数上不断做实验
   - latent pooling
   - layer fusion pooling
   - 多版本 hard-negative / MixCSE / Slerp 变体 loss

## 3. 代码入口与目录含义

### 3.1 模型核心目录

主要模型代码在：

- `llm2vec/`

其中最重要的几个文件：

- `llm2vec/llm2vec.py`
  - 当前默认导出的 `LLM2Vec`
  - 支持 `mean`、`weighted_mean`、`eos_token`、`last_token`、`bos_token`、`latent_pooling`
- `llm2vec/llm2vecV1.py`
  - 旧主线版本
  - 之前很多实验实际使用的是这个版本
- `llm2vec/llm2vecV3.py`
  - 在 V1 基础上增加了多层特征融合
  - 支持 `pooling_mode="layer_fusion"`

一个重要提醒：

- `llm2vec/__init__.py` 默认导出的是 `llm2vec.py` 里的 `LLM2Vec`
- 但 `experiments/run_supervised_with_eval.py` 明确 import 的是 `llm2vec.llm2vecV1`
- `experiments/run_supervised_fusion_withEval.py` 明确 import 的是 `llm2vec.llm2vecV3`

因此，比较实验结果时，必须同时记录：

- 训练脚本
- 实际用到的 `LLM2Vec` 版本
- pooling mode
- loss 版本

否则很容易出现“看起来只改了一个超参，实际底层类已经变了”的混淆。

### 3.2 latent pooling 相关

latent pooling 的实验代码主要在：

- `llm2vec/pooling_latent_V0.py`
- `llm2vec/pooling_latent_V1.py`
- `llm2vec/pooling_latent_V2.py`
- `llm2vec/pooling_latent_V3.py`
- `llm2vec/pooling_latent.py`

这些文件是在模仿 NV-Embed 的 latent attention pooling 思路做多轮演化：

- V0: 最基础 latent dictionary + attention + mean pool
- V1: 加 LayerNorm 和 residual
- V2: 更接近 NV-Embed 风格的 PreNorm + cross-attn + FFN + residual
- V3: 在 V2 上加强 backend 兼容性和细节稳定性
- `pooling_latent.py`: 当前更偏“可实际跑”的版本，初始化和 device/dtype 处理更保守

要注意的一点是：

- 文件很多，但运行时不一定真的用了你以为的那个版本
- 当前 `llm2vec.py` / `llm2vecV1.py` / `llm2vecV3.py` 导入的是 `llm2vec/pooling_latent.py`
- 所以做 latent pooling 对比时，最好在日志里显式记录实际模块来源

### 3.3 数据相关

数据集类主要在：

- `llm2vec/dataset/Derm1M.py`
- `llm2vec/dataset/Derm1M_SimVariants.py`
- `llm2vec/dataset/Derm1M_Variants_Eval.py`
- `llm2vec/dataset/DermVariants.py`
- `llm2vec/dataset/DermQA.py`

其中目前最关键的是：

- `Derm1M`
  - 更像领域文本预训练/MNTP 语料
- `DermVariants`
  - 当前监督微调主数据集
  - 是五个子任务的混合数据集

`DermVariants` 当前混合了以下子集：

- `SemVariants`
- `VisVariants`
- `DermQA`
- `SI1`
- `SI2`

它们共享一个对比学习训练框架，但任务定义并不完全一致。

### 3.4 训练脚本相关

当前值得优先记住的训练入口：

- `experiments/run_supervised_with_eval.py`
  - 监督微调主入口之一
  - 使用 `llm2vecV1`
- `experiments/run_supervised_fusion_withEval.py`
  - 融合层版本训练入口
  - 使用 `llm2vecV3`
- `experiments/run_supervised_FocalMixCSE.py`
  - 面向特定 loss 变体的训练入口

历史脚本和旧实验大多保存在：

- `experiments/Archive/`

## 4. 目前已经明确的项目难点

从 `experiments/retrospectives` 里的记录来看，这个项目目前的主要瓶颈并不一定是“模型还不够复杂”，而更可能是下面几类问题叠加：

### 4.1 数据目标不完全一致

`DermVariants` 是混合任务数据集，不是单一 retrieval 定义：

- 有的更像语义改写匹配
- 有的更像问答检索
- 有的更像诊断文本和视觉描述文本之间的对齐

这意味着模型被迫在一个 embedding 空间里同时兼顾多种相似性定义。

### 4.2 不同子集格式不统一

当前不同子集对 instruction 的使用方式不同：

- `SemVariants`、`VisVariants`、`SI1`：query / positive / negative 都带 instruction
- `DermQA`、`SI2`：通常只有 query 侧带 instruction

这会让模型学到部分“格式提示”而不完全是纯语义对齐。

### 4.3 hard negative 质量不稳定

从已有复盘看：

- `VisVariants` 的 hard negatives 常常非常接近正样本，甚至边界不清
- `SI1` 存在“positive 很短、negative 反而更完整”的结构性问题
- `DermQA` 和部分 `SI2` 里也会出现主题很近但标签定义不够干净的情况

因此，很多 loss 变体虽然形式更复杂，但底层仍继承了 hard negative 噪声。

### 4.4 架构改动可能被数据问题掩盖

目前已经试过的方向包括：

- latent pooling 多版本
- loss V0-V5 多版本
- layer fusion pooling
- 其他 MixCSE / Slerp / top-k shared negative 设计

但已有复盘倾向于说明：

- 如果主要瓶颈来自数据混合和负样本质量
- 单纯继续堆 pooling 或 loss 复杂度，收益可能非常有限

## 5. retrospectives 目录的用途

这个目录不是“成功实验展示”，而是项目记忆库，主要用于保存：

- 失败实验
- 原因分析
- 暂时证伪的方法
- 下次不要重复踩的坑

目前结构：

- `experiments/retrospectives/data/`
  - 数据质量、采样、负样本、分布、格式问题
- `experiments/retrospectives/methods/`
  - 模型结构、池化、loss、训练动态、验证行为

建议把未来每次实验都最少记录以下信息：

- 日期
- 训练脚本
- `LLM2Vec` 类版本
- pooling mode
- loss 版本
- 数据集及采样设置
- 核心超参
- 最重要的结果与失败现象
- 是否继续追这个方向

## 6. 已有复盘结论摘要

### 6.1 data 侧

`data/2026-03-18_data_first_analysis.md` 的核心结论：

- 当前训练集本质上是多任务混合
- 子集间 instruction 格式不统一
- 训练比例受 `SI1` / `SI2` 强影响
- 部分子集的 hard negative 可能过脏或过难
- 当前问题不应被直接判断为“模型容量不足”

### 6.2 methods 侧

`methods/` 下已有文档说明：

- `llm2vecV1` 与 `llm2vecV3` 是不同实验分支
- latent pooling 已经做过多代设计
- loss 从 V0 到 V5 已经有较多尝试
- layer fusion 也已经单独作为方向探索

一个非常重要的项目共识是：

> 下一轮迭代应更偏向 evidence-driven，而不是继续盲目扩展模型复杂度。

## 7. 下次登录新机器时建议先看什么

建议恢复上下文时按这个顺序读：

1. 本文件
2. `experiments/retrospectives/README.md`
3. `experiments/retrospectives/data/2026-03-18_data_first_analysis.md`
4. `experiments/retrospectives/methods/2026-03-18_methods_inventory.md`
5. 你这次准备使用的训练脚本
6. 对应脚本 import 的 `LLM2Vec` 版本文件

如果是准备继续做方法实验，优先再看：

- `experiments/retrospectives/methods/2026-03-18_latent_pooling_designs.md`
- `experiments/retrospectives/methods/2026-03-18_loss_versions_v0_to_v5.md`
- `experiments/retrospectives/methods/2026-03-18_layer_fusion_pooling.md`

## 8. 当前建议的工作原则

- 先锁一个稳定 baseline，再做新改动
- 所有实验先跑低成本短程筛选，再决定是否全量长训
- 每次实验必须记录脚本版本、模型类版本、loss、pooling 和数据配置
- 在继续改模型前，优先验证数据侧 margin、hard negative 质量和子集混合策略

---

这份文档的定位不是论文式总结，而是“项目交接说明 + 记忆恢复索引”。
下次换机器登录时，先读它，再决定进入数据线还是方法线，会更省时间。
