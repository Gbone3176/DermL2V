# MixCSE Fusion 模块核心设计摘要

本文件用于快速回忆 `layer_fusion` 的设计与训练要点，避免每次重新通读全项目代码。

## 1. 设计目标
- 在 `LLM2Vec` 中新增 `pooling_mode="layer_fusion"`，利用多层 hidden states 的信息做动态融合，增强句向量表示能力。
- 与原有 `mean/latent_pooling` 并存，可通过配置切换。

## 2. 模块结构（核心）
- 位置: `llm2vec/llm2vecV3.py` 中的 `LLM2Vec`。
- 新增配置参数:
  - `layer_fusion_num_layers`: 取最后 K 层参与融合。
  - `layer_fusion_temperature`: 路由 softmax 温度。
  - `layer_fusion_hidden_dim`: 路由 MLP 的中间维度。
  - `layer_fusion_train_only`: 是否仅训练 fusion 分支（冻结 backbone）。
  - `layer_fusion_gamma_init`: 融合残差系数初始值。
  - `layer_fusion_gamma_learnable`: gamma 是否可学习。
- 新增子模块:
  - `layer_fusion_norm = LayerNorm(hidden_size)`
  - `layer_fusion_router = Linear(hidden_size, router_hidden) -> GELU -> Linear(router_hidden, 1)`
  - `layer_fusion_gamma`（Parameter 或 buffer）
- 初始化策略:
  - router 最后一层权重/偏置置零，初期近似均匀路由，训练中再学习层权重。

## 3. Forward / Pooling 逻辑
- 当 `pooling_mode == "layer_fusion"` 时，前向会强制 `output_hidden_states=True`。
- 从 backbone 输出中取最后 K 层 hidden states（不含 embedding 层）。
- 对每层做 masked mean pooling，得到 `(B, K, D)`。
- 经 `LayerNorm + Router MLP` 计算每层分数，温度缩放后 softmax 得到层权重。
- 计算融合向量 `fused = sum_k(w_k * h_k)`。
- 基础向量 `base = mean_pool(last_hidden_state)`。
- 最终输出: `base + gamma * fused`（残差式融合）。

## 4. 训练策略（Fusion-only）
- 在训练脚本 `experiments/run_supervised_fusion_withEval.py` 中:
  - 若 `pooling_mode=="layer_fusion"` 且 `layer_fusion_train_only=true`:
    - 跳过 LoRA 初始化。
    - 调用 `freeze_backbone_for_fusion_training()` 冻结 backbone，仅训练:
      - `layer_fusion_router.*`
      - `layer_fusion_norm.*`
      - `layer_fusion_gamma`
    - 额外检查 backbone 是否仍有可训练参数，若有则报错。
- 会统计并打印 fusion/LoRA/latent 各分支可训练参数量（主进程打印）。

## 5. 保存与加载
- `LLM2Vec.save()` 会额外保存:
  - `llm2vec_config.json`（含 fusion 超参）
  - `layer_fusion_router.pt`（router + norm + gamma）
- `from_pretrained()` 会在 `pooling_mode=layer_fusion` 时尝试 `_load_layer_fusion_weights()`。

## 6. 参数使用注意
- `layer_fusion_hidden_dim` 不需要等于模型 `hidden_size`。
  - 它是 router MLP 中间层维度，不是 backbone 隐层维度。
  - 过大只会增加参数和显存。
- 修改 `layer_fusion_hidden_dim` 后，旧的 `layer_fusion_router.pt` 可能 shape 不匹配，需重新训练或不加载旧 router 权重。
- 你当前配置样例（8B）:
  - `pooling_mode=layer_fusion`
  - `layer_fusion_num_layers=4`
  - `layer_fusion_temperature=1.0`
  - `layer_fusion_hidden_dim=1024`
  - `layer_fusion_train_only=true`
  - `layer_fusion_gamma_init=1e-3`

## 7. 代码入口说明（避免混淆）
- 目前仓库中:
  - `llm2vec/llm2vec.py` 是历史版本（不含 fusion 逻辑）。
  - `llm2vec/llm2vecV3.py` 含 fusion 逻辑。
- 后续若要稳定复现实验，请确认训练入口使用的是带 fusion 的 `LLM2Vec` 实现。

