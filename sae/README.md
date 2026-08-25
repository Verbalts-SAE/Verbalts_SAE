# Pure Top-K SAE

One SAE variant is retained under `results/` and the pure baseline under
`artifacts/`. The SAE uses residual layer 1 of Pure VerbalTS, 64-dimensional
activations, a 512-dimensional dictionary, and Top-K=48:

- `artifacts/no_sae_pure_verbalts/verbalts.ckpt`: the no-SAE baseline.
- `results/sae_retrain/models/t5_45/`: SAE active for steps 5–45
  (`best.pt` plus `config.json` / `metrics.json`).
- `results/sae_retrain/classwise_classifiers/t5_45/best.pt`: position-aware
  3-by-4 segment-shape classifier trained on SAE latents.
- `results/sae_retrain/classwise_steering/evaluation_classifier/segment_cnn.pth`:
  held-out-tested 1D CNN used to score generated curves.

The former t=0–45 and t=40–45 SAE variants, all fixed-direction IG steering
weights, and every mechanism-analysis artifact have been removed.

## Source layout

Training pipeline (every stage writes provenance JSON):

- `collect_activations.py`: reproducibly collect residual-layer activations.
- `training/train_sae.py` + `training/train_t5_45.sh`: fixed t=5–45 recipe.
- `latent_classifier.py` + `train_latent_classifier.py`: position-aware
  3-by-4 segment-shape classifier on SAE latents.

Evaluation / visualization:

- `evaluate_compositional_full.py`: paired pure-vs-guidance generation and
  metrics (CTTP, MSE, segment CNN); entry point of the hybrid13 recipe.
- `evaluate_control_variants.py`: dense-vs-sparse guidance ablations; entry
  point of the `sparse_dynamic_k32_eta5120` recipe.
- `visualize_variants.py`: pure-vs-SAE reconstruction comparison figures.
- `diagnose_h_dynamics.py`: h-space classifier helper imported by the
  evaluation scripts.

Infrastructure:

- `wrappers.py`: attach SAEs / guidance wrappers to VerbalTS (including the
  per-token top-k dynamic guidance mode).
- `provenance.py`: SHA-256 + runtime metadata for every pipeline artifact.
- `contsg/models/sae_module.py`: the pure Top-K architecture.

Activation `.pt` files are caches and are intentionally not retained. To
retrain, collect the window first and then run the fixed training script.

## Final recipes

### hybrid13 — dense latent classifier guidance (retained winner)

Shape-aware hybrid guidance in the t=5–45 window (dense full-512-dim latent
gradient, gamma=6, eta=2560, full strength on "double peaks,sag"):

```bash
python -m sae.evaluate_compositional_full \
  --mode guidance --windows t5_45 --strengths 2560 \
  --guidance-adaptive 6.0 --guidance-full-strength "double peaks,sag" \
  --output-dir results/compositional_guidance_hybrid13 --device cuda
```

Key results vs Pure VerbalTS (n=4000, seed 42): segment accuracy +4.35pp
(0.7778→0.8213), whole-curve exact match +6.65pp (0.5048→0.5713), CTTP
46.34→46.92, MSE +11.0%. Multi-seed (42/7/11) Δseg = +4.37±0.15pp,
Δwhole = +6.74±0.07pp, McNemar p≈0. Details in
`results/compositional_guidance_hybrid_final.md`.

### sparse_dynamic_k32_eta5120 — sparse per-token top-32 guidance (candidate)

Same hybrid recipe but the guidance step is truncated to the top-32 largest
per-token latent moves each diffusion step (the other 480 dims stay frozen):

```bash
python -m sae.evaluate_control_variants \
  --mode sparse \
  --output-dir results/control_ablations/sparse --device cuda
```

The script runs all dense/global/dynamic top-k variants in one pass;
`sparse_dynamic_k32_eta5120` is its `LatentClassifierGuidanceWrapper`
configured with `topk=32, topk_mode="dynamic", eta=5120`, sharing the
hybrid gamma=6 full-strength shape schedule.

Key results vs Pure VerbalTS (n=256, seed 42): whole-curve exact match +6.3pp
(0.5234→0.5859), segment accuracy +4.0pp (0.7878→0.8281), MSE -0.7%
(1.0333→1.0266; paired 95% CI [-0.041, +0.027], no significant degradation).
It captures 73% of the dense ACCR gain at near-zero MSE cost. Full-scale
n=4000 multi-seed confirmation is pending.

## Comparing the retained variants

Run the reproducible synth-u visualization pipeline from the repository root:

```bash
python -m sae.visualize_variants
```

The pipeline generates a deterministic pool of 128 shape-rich test
conditions, scores the SAE reconstruction effect against Pure VerbalTS, and
selects six strong examples balanced between correct and wrong CNN-category
changes. Both variants use the same DDIM random stream for a fair
per-condition comparison.

These retained SAEs replace hidden states with their reconstructions in a
fixed diffusion-timestep window. They do **not** add or subtract a selected
latent feature direction, so the result is an SAE reconstruction intervention
rather than target-directed feature steering.

## Loading

```python
from sae.wrappers import attach_sae, detach_sae

# model is an already loaded VerbalTSModule.
attach_sae(model, "results/sae_retrain/models/t5_45/best.pt")

# Restore the Pure VerbalTS path.
detach_sae(model)
```

## Cleanup note

The following have been removed as superseded or transient:

- t=0–45 and t=40–45 SAE checkpoints and all experiments built on them
  (fixed-direction steering, single-neuron / group-level transport,
  linear/IG-direction classifiers, height-width and pems-sf generalization
  line, h-guidance and global-top-k ablation outputs, mechanism experiments
  p0–p8 and their diagnosis/analysis scripts);
- all `run_*.slurm` submission scripts and `slurm_log/` outputs;
- experiment logs under `real_dataset_annotations/amplitude_controls/`;
- the pems-sf-height-width dataset cache (289 MB) and its generator config.

Conclusions of the archived experiments remain documented in
`results/archived_experiments_summary.md`.
