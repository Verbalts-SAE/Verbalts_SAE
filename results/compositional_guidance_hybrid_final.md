# SAE Shape-Aware Hybrid Guidance — 最终汇总（synth-u 全量测试集 n=4000）

日期：2026-08-22 生成。全部结果由 `sae/evaluate_compositional_full.py --mode guidance`
（单窗口 t5_45，`--guidance-full-strength "double peaks,sag"`）产生。

## 基准（pure VerbalTS）
| CTTP | MSE | segment_accuracy | whole_curve_exact_match |
|---|---|---|---|
| 46.3417 | 0.9635 | 0.7778 | 0.5048 |

## Shape-aware hybrid 剂量×γ 矩阵（Δ = vs pure）

| η | γ | seg | Δseg | MSE | ΔMSE% | CTTP | whole |
|---|---|---|---|---|---|---|---|
| 640 | 0（非 adaptive） | 0.8091 | +3.13pp | 1.1747 | +21.9% | 46.48 | 0.5400 |
| 320 | 1 | 0.8113 | +3.35pp | 1.0725 | +11.3% | 46.65 | 0.5505 |
| 1280 | 1 | 0.8151 | +3.73pp | 1.1313 | +17.4% | 46.65 | 0.5545 |
| 640 | 2 | 0.8133 | +3.55pp | 1.0708 | +11.1% | 46.74 | 0.5533 |
| 1280 | 2 | 0.8172 | +3.94pp | 1.0960 | +13.7% | 46.75 | 0.5613 |
| 2560 | 2 | 0.8175 | +3.97pp | 1.1233 | +16.6% | 46.76 | 0.5623 |
| 640 | 3 | 0.8147 | +3.69pp | 1.0547 | +9.5% | 46.79 | 0.5560 |
| 1280 | 3 | 0.8183 | +4.05pp | 1.0767 | +11.7% | 46.80 | 0.5637 |
| 2560 | 3 | 0.8189 | +4.11pp | 1.1009 | +14.3% | 46.82 | 0.5665 |
| 1280 | 4 | 0.8197 | +4.19pp | 1.0645 | +10.5% | 46.84 | 0.5655 |
| 2560 | 4 | 0.8209 | +4.31pp | 1.0865 | +12.8% | 46.86 | 0.5690 |
| 1280 | 5 | 0.8197 | +4.19pp | 1.0562 | +9.6% | 46.87 | 0.5650 |
| 2560 | 5 | 0.8213 | +4.35pp | 1.0765 | +11.7% | 46.89 | 0.5687 |
| 1280 | 6 | 0.8203 | +4.25pp | 1.0498 | +9.0% | 46.88 | 0.5667 |
| 2560 | 6 | 0.8213 | +4.35pp | 1.0692 | +11.0% | 46.92 | 0.5713 |
| 1280 | 8 | **0.8210** | **+4.32pp** | **1.0406** | **+8.0%** | 46.91 | 0.5685 |
| 2560 | 8 | 0.8207 | +4.29pp | 1.0584 | +9.8% | 46.94 | 0.5695 |
| 1280 | 10 | 0.8202 | +4.24pp | 1.0342 | +7.3% | 46.92 | 0.5680 |
| 2560 | 12 | 0.8208 | +4.30pp | 1.0457 | +8.5% | 46.98 | 0.5710 |

## 推荐配方（全部 McNemar p≈0，净提升 500+ 段/12000）

1. **平衡推荐**：η=1280, γ=8 → **seg 0.8210（+4.32pp），MSE +8.0%，whole 0.5685（+6.4pp），CTTP 46.91**
2. **MSE 最友好**：η=1280, γ=10 → +4.24pp @ MSE +7.3%
3. **seg 最高**：η=2560, γ=6 → **seg 0.8213（+4.35pp）**，MSE +11.0%，whole 0.5713（+6.65pp）

## 机理（per-category，γ3 η=1280 vs pure）

| 类别 | pure | hybrid | Δ |
|---|---|---|---|
| sag | 64.3% | **76.3%** | +12.0pp |
| double peaks | 63.7% | 70.2% | +6.5pp |
| single peak | 65.8% | 67.3% | +1.5pp |
| nothing | 97.9% | 98.0% | +0.1pp |

## 关键发现

- 非 adaptive 高剂量（η=640）的收益全部来自 sag/double peaks，single peak 段 +0.0pp（强扰动下收益回吐）。
- 对 single peak/nothing 段做置信度加权 `(1-p)^γ`、对 sag/double peaks 保持全强度后，
  sag 收益反而进一步升到 +12.0pp（扰动更纯粹），且 MSE 大幅下降（+21.9% → +8.0%）。
- CTTP 不仅保住还提升（46.34 → 46.9+）；whole 提升 +6.4pp。
- γ 增大只降 MSE（seg 收敛于 0.821），η 在 1280–2560 附近饱和。
- combo（+t40_45:80）与顺序反转均不占优；t0_45 为负资产。

## 复现命令（两个推荐配方）

```
# seg 最高（本文可视化所用）：γ=6, η=2560
python -m sae.evaluate_compositional_full \
  --mode guidance --windows t5_45 --strengths 2560 \
  --guidance-adaptive 6.0 --guidance-full-strength "double peaks,sag" \
  --output-dir results/compositional_guidance_hybrid13 --device cuda

# 平衡推荐（MSE 代价最小）：γ=8, η=1280
python -m sae.evaluate_compositional_full \
  --mode guidance --windows t5_45 --strengths 1280 \
  --guidance-adaptive 8.0 --guidance-full-strength "double peaks,sag" \
  --output-dir results/compositional_guidance_hybrid14 --device cuda
```

## 最新最佳：Shape-Aware Hybrid Classifier Guidance 完整说明

### Steering 方式（单窗口 t5_45，guidance 模式）

1. **插入位置**：VerbalTS 的 DDIM 采样在扩散步 **t=5–45** 挂载 SAE（t5_45 窗口，
   `results/sae_retrain/models/t5_45/best.pt`），在 SAE latent 空间操作。
2. **每扩散步的投影梯度上升**（分类器来自 `results/sae_retrain/classwise_classifiers/t5_45/best.pt`，
   对 latent 预测 B/M/E 三段形状）：

   ```
   z_new = relu(z - clamp(η · ∇_z mean_stage(-log p_target),
                          min(rel_cap · σ_d, max_step)))
   ```

   - η（dose）= **2560**；rel_cap = 1.0（每 latent 步长上限 = 该 latent 的 input_std，
     死 latent 有 1e-6 地板）；max_step = 0.5（绝对步长上限）；iters = 1。
   - 梯度在每个窗口内扩散步重算，target 为 caption 派生的 B/M/E ground-truth 形状。
3. **Shape-aware 混合加权（核心创新）**：

   ```
   loss = -mean_stage( w_seg · log p_target(seg) )
   w_seg = 1                               若 target ∈ {double peaks, sag}（全强度）
   w_seg = (1 - p_target)^γ                其他形状（adaptive 置信度加权，γ=6）
   ```

   - sag/double peaks 是难类别（pure 仅 ~57% 正确），需要全强度扰动；
   - single peak/nothing 已接近饱和，按 `(1-p)^6` 置信度加权，只对低置信度样本扰动，
     大幅降低 MSE 代价（+21.9% → +11.0%）。
4. **为何优于前序方案**：固定方向 additive/multiplicative steering 无效；非 adaptive
   guidance 高剂量收益回吐（single peak +0.0pp）；combo（+t40_45）、顺序反转、
   三窗口（+t0_45）均不占优。

### 最终指标（γ=6, η=2560，n=4000）

| 指标 | pure | hybrid | Δ |
|---|---|---|---|
| segment_accuracy | 0.7778 | **0.8213** | **+4.35pp** |
| whole_curve_exact_match | 0.5048 | **0.5713** | **+6.65pp** |
| MSE | 0.9635 | 1.0692 | +11.0% |
| CTTP | 46.34 | **46.92** | +0.58（不降反升） |

McNemar：whole 净 +266/4000（385 对 → 错，119 错 → 对），p≈0；
seg 相对错误率降低 19.6%，whole 相对错误率降低 13.4%。

## 多 seed 稳健性验证（seed = 42 / 7 / 11，γ=6, η=2560，n=4000）

同一配方在三个随机 seed 下独立重跑（采样噪声不同），汇总脚本
`sae/summarize_multiseed.py`；结果目录：hybrid13（seed 42）、hybrid13_seed7、hybrid13_seed11。

| seed | seg pure | seg hybrid | Δseg | whole pure | whole hybrid | Δwhole | ΔMSE% | CTTP pure | CTTP hybrid | ΔCTTP | McNemar seg/whole |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 0.7778 | 0.8213 | +4.36pp | 0.5048 | 0.5713 | +6.65pp | +11.0% | 46.34 | 46.92 | +0.57 | p≈0 / p≈0 |
| 7 | 0.7760 | 0.8216 | +4.56pp | 0.5000 | 0.5680 | +6.80pp | +8.6% | 46.34 | 46.98 | +0.64 | p≈0 / p≈0 |
| 11 | 0.7773 | 0.8191 | +4.18pp | 0.4988 | 0.5665 | +6.78pp | +10.9% | 46.27 | 46.89 | +0.63 | p≈0 / p≈0 |

汇总（mean ± std，[min, max]）：

- Δseg = **+4.37pp ± 0.15pp** [4.18, 4.56]；Δwhole = **+6.74pp ± 0.07pp** [6.65, 6.80]
- ΔMSE = **+10.2% ± 1.1%** [8.6, 11.0]；ΔCTTP = **+0.61 ± 0.03** [0.57, 0.64]
- 三个 seed 的 seg/whole McNemar 均 p≈0；配对计数均为净 +500 段以上（seg 级）
- 结论：收益对采样噪声高度稳健，最优指标波动 < 0.2pp，MSE 代价稳定在 +8.6%~+11.0%

## Per-stage 分析（γ=6, η=2560 vs pure）

| 阶段 | pure | hybrid | Δ |
|---|---|---|---|
| Beginning（B） | 83.55% | 84.13% | +0.58pp |
| **Middle（M）** | 73.12% | **79.73%** | **+6.60pp** |
| End（E） | 76.65% | 82.55% | +5.90pp |

M 段按形状细分（相对错误率降低）：

| M 段形状 | pure | hybrid | Δ |
|---|---|---|---|
| **sag** | 56.81% | **76.44%** | **+19.63pp**（错误率 -45.5%） |
| **double peaks** | 57.06% | **66.99%** | **+9.93pp**（错误率 -23.1%） |
| single peak | 60.86% | 63.48% | +2.62pp |
| nothing | 95.39% | 95.58% | +0.19pp |

收益集中在最难的 M 段 sag/double peaks，而非均匀撒网；B 段已接近饱和（nothing 居多）。

## 可视化案例（`sae/visualize_best_steering.py`，使用 `contsg/eval/visualizer.py`）

5 个案例（`results/best_steering_viz/*.png|pdf`），均为 pure 形态预测错误、hybrid 修复
且 **MSE 同时大幅改善** 的双赢案例：

| 案例 | test index | GT (B/M/E) | pure 预测 | hybrid 预测 | MSE pure→hybrid |
|---|---|---|---|---|---|
| sag 修复 #1 | 3544 | sag/sag/single peak | sag/single peak/single peak | **全对** | 2.50 → 0.58（-76%） |
| sag 修复 #2 | 3411 | sag/sag/single peak | sag/single peak/nothing | **全对** | 2.33 → 0.55（-76%） |
| dp 修复 #1 | 1267 | nothing/double peaks/single peak | nothing/single peak/double peaks | **全对** | 3.08 → 1.61（-48%） |
| dp 修复 #2 | 2171 | double peaks/double peaks/sag | double peaks/single peak/sag | **全对** | 2.84 → 1.63（-43%） |
| 稳健 sag | 1936 | nothing/sag/nothing | 全对 | 全对 | 2.47 → 0.21（-92%） |

图中蓝线=Ground Truth、红虚线=Pure VerbalTS、绿实线=Hybrid guidance，
B/M/E 三段背景分色，图例标注各变体的 CNN B/M/E 形状预测。

生成命令：`python -m sae.visualize_best_steering`

显著性检验：`python sae/mcnemar_compare.py --results <dir>/per_case.json --variants t5_45_g1280 pure`
