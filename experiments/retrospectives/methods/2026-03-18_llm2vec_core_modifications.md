# LLM2Vec 核心改动回顾

## 背景
- 目标：扩展基础 LLM2Vec，使其支持 instruction-aware masking、latent pooling 和更多 pooling 模式。
- 主要代码：
  - `llm2vec/llm2vecV1.py`
  - `llm2vec/llm2vecV3.py`
  - `experiments/run_supervised_with_eval.py`
  - `experiments/run_supervised_fusion_withEval.py`

## 做了什么改动
- `run_supervised_with_eval.py` 使用的 V1 分支导入了 `llm2vec.llm2vecV1.LLM2Vec`。
- V1 支持 `mean/weighted_mean/eos/last/bos/latent_pooling`，并包含 `skip_instruction` 和 `embed_mask` 流程。
- V3 分支新增了 `pooling_mode="layer_fusion"` 以及 fusion 专用的可训练模块。

## 为什么它可能失败
- 实现分支不一致有风险：不同脚本使用了不同版本的 LLM2Vec 类。
- 如果一个实验跑在 V1，另一个跑在 V3，方法比较很容易被实现差异污染。
- pooling 行为和 instruction masking 与 separator 格式强耦合；一旦数据格式漂移，行为会静默变化。

## 证据锚点
- `run_supervised_with_eval.py` 导入的是 `llm2vecV1`。
- `run_supervised_fusion_withEval.py` 导入的是 `llm2vecV3`。
- `llm2vec/__init__.py` 的默认导出指向 `llm2vec.py`，不是 V1。

## 结论
- 两个分支都可以保留，但实验元数据里必须始终记录脚本路径和类版本。

## 下一步检查
- [ ] 在运行头信息中打印：LLM2Vec 类路径 + git commit + loss 类名。
- [ ] 在配置中加保护，避免未显式声明时把 V1 和 V3 的结果混着比较。
