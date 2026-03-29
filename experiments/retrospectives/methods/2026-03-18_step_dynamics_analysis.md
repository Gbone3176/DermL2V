# Step 动态分析

## 背景
- 运行系列：`withEval_QAx10_SlerpMixCSE_DermData2`
- 相关配置：
  - `train_configs/supervised/MetaLlama3.1_8B_inst-mntp_supervised@DermVariantsSFT.json`
- 相关 trainer state：
  - `output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10_SlerpMixCSE_DermData2/.../checkpoint-90/trainer_state.json`
- 分析脚本：
  - `experiments/retrospectives/data/analyze_step_dynamics.py`

## 重要修正
- 之前关于 epoch 边界的解释适用于 debug 配置，当时参数是：
  - `per_device_train_batch_size = 64`
  - `gradient_accumulation_steps = 8`
- 当前这次运行不一样：
  - `per_device_train_batch_size = 16`
  - `gradient_accumulation_steps = 4`
- 因此：
  - 每个 optimizer step 的有效样本数 = `16 * 4 * 4 = 256`
  - 对齐 batch 后的有效训练集大小 = `136960`
  - 每个 epoch 的 step 数 = `136960 / 256 = 535`
- 所以，这次运行中的 `step 55-85` 并不靠近 epoch 边界。

## 分析了什么
- epoch 边界标记
- step 级任务组成
- step 级平均 `pos_score - neg_score`
- 记录下来的 `grad_norm`

## 结果

### 1. step 50 到 90 附近的局部下滑不是由 epoch rollover 引起的
- 在这次运行里，epoch 边界在 `step 535`，不是 `67`。
- 因此，`step 55-85` 附近的局部行为一定另有原因。

### 2. 下滑窗口里的任务组成比较稳定
- 在 `45-95` 这些 step 之间，各子集占比基本维持在相近范围：
  - `SemVariants`：约 `0.08 - 0.18`
  - `VisVariants`：约 `0.13 - 0.27`
  - `DermQA`：约 `0.04 - 0.11`
  - `SI1`：约 `0.23 - 0.37`
  - `SI2`：约 `0.26 - 0.37`
- 没有看到明显的子集切换或主导权翻转。

### 3. 平均 margin 也相对稳定
- 在同一窗口内，step 级平均 margin 大多在：
  - `0.12 - 0.15`
- 虽然有波动，但没有与指标下滑精确对应的明显断崖。

### 4. `grad_norm` 先较早触底，然后缓慢回升
- 窗口内记录值：
  - step `45`：约 `0.918`
  - step `55`：约 `0.802`
  - step `65`：约 `0.800`
  - step `75`：约 `0.840`
  - step `85`：约 `0.891`
- 这更像是：
  - 早期快速稳定
  - 然后进入低梯度平台
  - 接着更新幅度轻微再扩张

## 当前解释
- 这次下滑不太可能由以下因素导致：
  - epoch 边界
  - 子集组成的突然变化
  - 平均 hard-negative margin 的突然下降
- 它更可能与优化行为有关，例如：
  - 先经过一个容易拟合的阶段，随后进入更难的局部调整
  - 表示几何先发生变化，而下游检索指标稍后才恢复
  - 训练 loss 下降与检索验证质量之间存在局部不匹配

## 实际启示
- 对这次运行来说，“数据组成冲击”不是解释 `50-90` 下滑的主因。
- 更可能的解释是：
  - 混合任务训练下的优化阶段动态
  - 再叠加 noisy 或 near-boundary 的 hard negative

## 生成产物
- `experiments/retrospectives/data/step_dynamics_outputs/step_dynamics_full.csv`
- `experiments/retrospectives/data/step_dynamics_outputs/focus_steps_45_95.csv`
- `experiments/retrospectives/data/step_dynamics_outputs/step_dynamics_overview.png`

## 下一步动作
- [ ] 把 step 级验证指标和同一时间轴上的训练动态画到一张图里
- [ ] 检查是否是某一个下游子集驱动了这次下滑，而不是整个验证集一起下滑
- [ ] 增加 step 级低 margin 比例，而不只看平均 margin
- [ ] 把这次运行与 debug 配置做对比，区分 batch-size 效应和数据顺序效应
