# 归档实验结论总结（2026-08-22 清理时记录）

> 本文件记录 `results/` 中被删除的中间探索实验的关键结论。
> 所有实验均为 synth-u 全量测试集 n=4000（smoke 为子集），pure VerbalTS 基线：
> **segment_accuracy=0.7778，MSE=0.9635，whole=0.5048，CTTP=46.34**。
> 最终胜出方案见 `compositional_guidance_hybrid_final.md`（shape-aware hybrid classifier guidance，γ=6 η=2560，t5_45 单窗口）。

## 一、Additive 固定方向 steering 系列（全部无效，已弃用）

| 目录 | 配置 | best seg（Δ vs pure） | ΔMSE | 结论 |
|---|---|---|---|---|
| compositional_additive_sweep | s∈{0.25,0.5,1,2} × {t0_45,t40_45,t5_45} | 0.7777（-0.01pp） | +3.4% | IG 固定方向加性干预跨窗口全无效 |
| compositional_add_rms_boost | RMS 校准 boost s∈{0.1,0.3,1} | 0.7775（-0.03pp） | +4.6% | 无效 |
| compositional_add_rms_boost48 | topk48 SAE，s∈{0.05,0.1,0.3} | 0.7778（+0.01pp） | +1.7% | 无效 |
| compositional_add_rms_both | 双向 boost+suppress | 0.7774（-0.03pp） | +2.2% | 无效 |
| compositional_add_rms_fine_boost | 细剂量 s∈{0.01,0.02,0.05} | 0.7782（+0.05pp） | +0.1% | 剂量降到近零也无效 |
| compositional_add_rms_fine_both | 细剂量双向 | 0.7781（+0.03pp） | -0.1% | 无效 |
| compositional_full_eval_t5_45 | 乘性/大剂量加性 s∈{2,4,8,12} | 0.7587（-1.91pp） | +49.8% | 大剂量强破坏 |
| compositional_full_eval | 乘性 ×2/×0.2 | 0.7616（-1.62pp） | +43.8% | 乘性干预全面恶化 |

**核心结论**：IG 归因向量是全局平均贡献，不是 per-sample 最优方向；任何固定方向
steering（加性/乘性/boost/suppress/细剂量）在生成级均无正向收益。这也是后续转向
per-sample classifier guidance 的根本原因。

## 二、早期因果验证系列（Exp1-4，探针级/生成级，已记录结论）

| 目录 | 内容 | 结论 |
|---|---|---|
| sae_single_neuron_transport | 单神经元运输（t40_45/t0_45） | 单神经元干预效果有限，升级为 group-level |
| sae_group_transport | Group-level 因果运输 + Exp4 hit-rate | 67% hit rate 无法被 strength-1 的 t40_45/t0_45 SAE/factor 对复现（location-only 0%）；sensitivity 到 strength 16 也仅边缘提升 |
| sae_factor_composition | 因子方向组合运输 | 因子方向运输同样无法稳定复现目标命中 |
| sae_variant_visualization | SAE 变体对比图（保留） | 唯一保留的早期分析可视化 |

## 三、Guidance 早期探索（γ 前身：非 adaptive / adaptive / combo / 窗口探索）

| 目录 | 配置 | best seg（Δ vs pure） | ΔMSE | 结论 |
|---|---|---|---|---|
| compositional_guidance_sweep | η∈{0.5,2,5,10} t40_45+t5_45 | 0.7844（+0.67pp） | +3.9% | 低剂量几乎无效果 |
| compositional_guidance_peak | η∈{80,160} | 0.8039（+2.62pp） | +13.8% | 高剂量开始见效但 MSE 代价大 |
| compositional_guidance_peak2 | η∈{320,640} 非 adaptive | 0.8091（+3.13pp） | +21.9% | 非 adaptive 最佳；收益全在 sag/dp，single peak +0.0pp |
| compositional_guidance_dose2 | η∈{20,40} 双窗口 | 0.7928（+1.51pp） | +7.9% | 中剂量不占优 |
| compositional_guidance_cap025 | rel_cap=0.25 η∈{20,40} | 0.7917（+1.39pp） | +7.7% | 缩小步长上限反而降低收益 |
| compositional_guidance_iters3 | iters=3 η∈{0.5,2} | 0.7791（+0.13pp） | +2.5% | 多步迭代不占优 |
| compositional_guidance_t0_45 | t0_45 窗口 η∈{0.2,0.5,2} | 0.7810（+0.32pp） | +1.2% | **t0_45 为负资产** |
| compositional_guidance_t0_45_dose2 | t0_45 η∈{5,10,20} | 0.7894（+1.17pp） | +4.7% | 提高剂量 t0_45 仍弱于 t5_45 |
| compositional_guidance_combo | combo 三窗口 | 0.7896（+1.18pp） | +3.2% | combo 不占优 |
| compositional_guidance_combo2 | combo 双窗口 | 0.7979（+2.02pp） | +9.1% | combo 不占优 |
| compositional_guidance_combo3 | combo 多规格 | 0.8060（+2.82pp） | +17.3% | combo 不占优（< 单窗口 0.8213） |
| compositional_guidance_adapt1 | γ=1 η∈{80,160,320} | 0.8021（+2.43pp） | +11.3% | adaptive 引入 |
| compositional_guidance_adapt2 | γ=1 combo | 0.7987（+2.10pp） | +11.2% | combo 弱 |
| compositional_guidance_adapt3 | γ=2 η∈{160,320} | 0.7997（+2.19pp） | +8.5% | 弱 |
| compositional_guidance_adapt4 | γ=0.5 η∈{320,640} | 0.8049（+2.72pp） | +17.1% | 弱 |
| compositional_guidance_adapt5 | γ=1 η∈{640,1280} | 0.8042（+2.64pp） | +17.5% | 弱 |
| compositional_guidance_adapt6 | γ=2 η∈{640,1280} | 0.8003（+2.26pp） | +10.9% | 弱 |
| compositional_guidance_adapt7 | γ=1 combo | 0.8003（+2.25pp） | +14.4% | combo 弱 |

**核心结论**：
- 非 adaptive 高剂量收益集中在 sag/double peaks（+8.2/+6.7pp），single peak 收益回吐（+0.0pp），MSE 代价大（+21.9%）。
- 全类别 adaptive `(1-p)^γ` 能恢复 single peak（+2.7pp），但牺牲 sag/dp 收益（降到 +3.5/+4.3pp）——单一策略存在本质 trade-off，催生 shape-aware hybrid。
- 窗口选择：t0_45 负资产；combo（t5_45+t40_45 加权）永远不占优；**t5_45 单窗口最优**。

## 四、Shape-aware Hybrid 剂量×γ 矩阵（中间点，最优已保留）

| 目录 | γ | η | best seg（Δ vs pure） | ΔMSE | 备注 |
|---|---|---|---|---|---|
| hybrid1 | 1 | 160/320/640 | 0.8113（+3.35pp） | +11.3% | 矩阵起点 |
| hybrid2 | 2 | 320/640 | 0.8133（+3.55pp） | +11.1% | |
| hybrid3 | 1 | 640/1280 | 0.8151（+3.73pp） | +17.4% | |
| hybrid4 | 2 | 1280/2560 | 0.8175（+3.97pp） | +16.6% | |
| hybrid5 | 3 | 640/1280 | 0.8183（+4.06pp） | +11.8% | |
| hybrid6 | 2 | combo | 0.8137（+3.60pp） | +12.7% | combo 弱 |
| hybrid7 | 3 | 2560 | 0.8189（+4.12pp） | +14.3% | |
| hybrid8 | 4 | 1280/2560 | 0.8209（+4.32pp） | +12.8% | |
| hybrid9 | 5 | 1280 | 0.8197（+4.19pp） | +9.6% | |
| hybrid10 | 5 | 2560 | 0.8213（+4.36pp） | +11.7% | |
| hybrid11 | 6 | 1280 | 0.8203（+4.26pp） | +9.0% | |
| hybrid12 | 5 | combo | 0.8168（+3.91pp） | +10.4% | combo 弱 |
| hybrid15 | 8 | 2560 | 0.8207（+4.29pp） | +9.8% | |
| hybrid17 | 12 | 2560 | 0.8208（+4.31pp） | +8.5% | γ 增大只降 MSE |

**核心结论**：γ 增大只降 MSE（seg 收敛于 ~0.821）；η 在 1280–2560 饱和。
Pareto 前沿三点：γ=6 η=2560（+4.35pp @ +11.0%，已保留为 hybrid13+多seed）、
γ=8 η=1280（+4.32pp @ +8.0%，保留 hybrid14）、γ=10 η=1280（+4.24pp @ +7.3%，保留 hybrid16）。

## 五、Smoke 测试（子集验证，无独立结论）

compositional_additive_sweep_smoke、compositional_guidance_smoke、compositional_guidance_combo_smoke、
compositional_guidance_adapt_smoke、compositional_full_eval_smoke、guidance_smoke：
均为小样本冒烟测试（n≈96-256），仅用于验证 pipeline 可运行，无统计效力，结论同对应全量实验。

## 六、全局结论（历次探索的沉淀）

1. **固定方向 steering（IG 归因向量）在生成级永远无效**——IG 是"全局平均地图"，不是 per-sample 导航方向；正确做法是 per-sample per-step 的 ∇_z log p_target（classifier guidance）。
2. **窗口位置不是关键杠杆**：v2 真实 t 下 20-25/30-35/40-45 各窗口 forcing 全部无效（-0.008~-0.014）；t0_45 在 guidance 范式下为负资产；combo 永远不占优。干预范式才是决定性因素。
3. **剂量策略必须 shape-aware**：全强度只给难类别（sag/double peaks），易类别按 `(1-p)^γ` 置信度加权，才能同时拿到最大 seg 增益和最小 MSE 代价。
4. 被删目录的关键原始数据不再保留；如需复核任何单点结论，可凭本表配置 + 代码重跑复现。

## 七、代码与日志清理记录（2026-08-22 第二轮）

按"早期脚本/slurm/log 不保留、只留结果总结"的要求，本轮清理如下。

**已删除的 Python 脚本（9 个，均为已弃用范式的实现/复现/诊断/可视化）：**
- `compositional_steering.py`：早期固定方向乘性/加性 steering 实现
- `run_steering.py`：class-wise IG Top-5/10 固定方向 steering 评估
- `diagnose_latent_shift.py`：additive 方向 latent 侧诊断
- `diagnose_guidance_dose.py`：guidance 剂量诊断
- `reproduce_single_neuron_transport.py` / `reproduce_group_transport.py` / `reproduce_factor_composition.py`：Exp1-4 因果运输复现
- `visualize_multiplicative_steering.py`：乘性干预可视化
- `visualize_classwise_steering.py`：class-wise 固定方向 steering 可视化

**已删除的 SLURM 脚本（49 个）：** 对应上表所有已删实验的提交脚本（additive/add_rms/boost/fine_dose/combo/adapt/dose/cap/peak/sweep/smoke/t0_45/hybrid1-12,15,17 等）。

**已删除的日志与早期产出：**
- `slurm_log/` 全部 70 个作业日志（已完结作业，结论均在各自 summary.json 与本表中）
- `results/sae_retrain/classwise_steering/{t40_45,t0_45,case_visualizations}`：固定方向 steering 的曲线与可视化
- `results/sae_retrain/multiplicative_steering_demo`：乘性干预 demo

**保留（当前 pipeline 硬依赖）：**
- 核心脚本：`wrappers.py`、`evaluate_compositional_full.py`、`latent_classifier.py`、`train_latent_classifier.py`、`attribute_latents.py`、`collect_activations.py`、`mcnemar_compare.py`、`summarize_multiseed.py`、`visualize_best_steering.py`、`visualize_variants.py`、`provenance.py`、`download_cttp_resources.py`、`training/`
- 复现脚本：`run_guidance_hybrid13[_seed7|_seed11].slurm`、`run_guidance_hybrid14.slurm`、`run_guidance_hybrid16.slurm`
- 资产 pipeline：`run_retrain_pipeline.slurm`、`run_classwise_pipeline.slurm`（已移除 run_steering 步骤）、`run_t5_45_pipeline.slurm`
- 评估 CNN：`results/sae_retrain/classwise_steering/evaluation_classifier/segment_cnn.pth`（evaluate_compositional_full.py 硬依赖）

## 八、代码精简与一致性修复（2026-08-22 第三轮）

针对 `sae/` 代码审查发现的问题，本轮修复如下（不改变任何实验结论）。

**主评估脚本精简（`evaluate_compositional_full.py` 862→689 行）：**
- 修复上轮遗漏的 ImportError：删除对已删模块 `compositional_steering` / `run_steering` 的导入，`generate_variant` 按 pyc 字节码恢复的原实现内联回主脚本（`model.generate(condition, n_samples=1, sampler="ddim")`，DDIM 流与纯基线配对）。
- 删除 3 个失败范式分支（additive / multiplicative / guidance_combo），`--mode` 仅保留 `guidance`；删除与之相关的组合参数（`--combo-specs`、`--beta`、`--gamma`、`--n-boost`、`--n-suppress`、`--classifier-epochs`、`--num-workers`）。
- `SHAPE_NAMES`/`STAGE_NAMES`/`SHAPE_TO_TARGET` 改从 `visualize_variants` 导入，消除重复定义。

**死代码清理：**
- `wrappers.py` 611→322 行：删除 5 个死类（LatentSteeringWrapper、MultiplicativeLatentSteeringWrapper、LatentPathWrapper、BatchConditionalAdditive/MultiplicativeSteeringWrapper），`__all__` 与 `sae/__init__.py` 导出同步更新。
- `mcnemar_compare.py`：删除死导入 `from math import isqrt`。
- `sae/__pycache__/`：删除 11 个死 `.pyc`（已删模块 ×7 + 旧 Python 3.8 缓存 ×4）。
- `tests/`：删除整体依赖已删模块的 `test_compositional_steering.py`、`test_sae_factor_composition.py`；`test_sae_visualization.py` 删 4 个死测试 + 2 行死导入；`test_sae_attribution.py` 删 5 个死测试 + 死导入与 `_identity_sae`。剩余 16 个 sae 相关测试全部通过。

**SLURM 一致性修复：**
- `run_t5_45_pipeline.slurm` 第 5 步：由过时的 `--windows t40_45,t5_45`（默认 additive 模式）改为最终方案命令（`--mode guidance --windows t5_45 --strengths 2560 --guidance-adaptive 6.0 --guidance-full-strength "double peaks,sag"`），输出到独立目录 `results/compositional_guidance_t5_45_pipeline`（避免覆盖已有 hybrid13 实验证据）。
- 5 个 guidance slurm（hybrid13/14/16/seed7/seed11）：补充 `mkdir -p slurm_log`，并将 hybrid13/14/16 头注释改为与命令实际一致的最终/消融描述。

**文档同步（`sae/README.md`）：**
- Source layout 按 pipeline 阶段重写（训练/归因 → 评估/可视化 → 基础设施），补上 `train_t5_45.sh`、`latent_classifier`、`attribute_latents` 等缺漏条目。
- 明确复现切入点：端到端 `sbatch sae/run_t5_45_pipeline.slurm`；已有模型时直接 `run_guidance_hybrid13[_seed7|_seed11]/14/16.slurm`。
- 清除中英混排注释（γ/η → gamma/eta，中文括号注记 → 英文）。

**验证：** `sae` 全部 13 个模块导入通过；全仓库 grep 无任何对已删模块/类的残留引用；pytest 16 passed。

