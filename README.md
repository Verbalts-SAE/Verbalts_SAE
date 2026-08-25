# VerbalTS-SAE

Sparse-autoencoder (SAE) latent steering for fine-grained morphological control
of text-conditioned time-series generation, built on **VerbalTS** and evaluated
on the **synth-u** subset of [ConTSG-Bench](https://github.com/seqml/ConTSG-Bench).

We insert a Top-K sparse autoencoder (64-dim residual stream → 512-dim latent
dictionary, Top-K = 48) into the residual bottleneck of VerbalTS. A
shape-aware classifier-guidance operator on the SAE latent space then edits the
generation process toward target morphologies (single peak / double peaks /
sag / nothing) per segment (beginning / middle / end).

## Results (synth-u, held-out CNN evaluator)

| Recipe | Description | Segment acc. | Whole-curve match | MSE |
| --- | --- | --- | --- | --- |
| Pure VerbalTS | no intervention | 0.7778 | 0.5048 | 1.0333 |
| `hybrid13` (dense γ=6, η=2560) | dense full-512-dim per-step gradient | **0.8213 (+4.35pp)** | **0.5713 (+6.65pp)** | +11.0% |
| `sparse_dynamic_k32_eta5120` | per-token top-32 sparse gradient | 0.8281 (n=256) | 0.5859 (n=256, +6.3pp) | **-0.7%** (no degradation) |

The sparse recipe truncates the guidance step to the top-32 largest per-token
latent moves each diffusion step, keeping the other 480 dimensions frozen. It
captures ~73% of the dense whole-curve gain at essentially zero MSE cost.
Full results and mechanism analysis:
[`results/compositional_guidance_hybrid_final.md`](results/compositional_guidance_hybrid_final.md);
archived negative results: [`results/archived_experiments_summary.md`](results/archived_experiments_summary.md).

## Installation

```bash
# Python 3.10+ with CUDA-enabled PyTorch is expected
pip install -e ".[full]"
# or, with uv:
uv sync
```

## Downloading data & models

The repository intentionally does **not** ship dataset files, CLIP weights, or
the 600 MB VerbalTS checkpoint. Download them once into the expected layout:

```text
Verbalts_SAE/
├── datasets/
│   └── synth-u/                  # 196 MB, ~16 npy/json files
├── checkpoints/
│   ├── longclip/                 # LongCLIP-GmP-ViT-L-14 (text encoder)
│   └── cttp/synth-u/             # released CTTP runtime files (2 files)
└── artifacts/
    └── no_sae_pure_verbalts/
        └── verbalts.ckpt         # 600 MB base generator
```

### 1. synth-u dataset

Released as part of the official ConTSG-Bench public dataset subset:

```bash
mkdir -p datasets
git lfs install
git clone https://huggingface.co/datasets/mldi-lab/ConTSG-Bench-Datasets datasets/tmp
# the repo contains synth-u/ (and synth-m/); keep only synth-u:
mv datasets/tmp/synth-u datasets/synth-u
rm -rf datasets/tmp
```

Or download directly from the HF dataset page:
https://huggingface.co/datasets/mldi-lab/ConTSG-Bench-Datasets

`datasets/synth-u/` contains `meta.json`, `train/valid/test_ts.npy`,
caption tensors, attribute indices, and precomputed Qwen3 text embeddings used
by the pipeline.

### 2. LongCLIP (CLIP) text encoder

The CTTP evaluation uses the **GmP fine-tuned LongCLIP** variant. Do **not**
substitute the original `BeichenZhang/LongCLIP-L`; released CTTP resources were
checked against the GmP variant.

```bash
mkdir -p checkpoints
git lfs install
git clone https://huggingface.co/zer0int/LongCLIP-GmP-ViT-L-14 checkpoints/longclip
```

### 3. Released CTTP resources (synth-u)

CTTP checkpoints and runtime configs required by generator evaluation:

```bash
mkdir -p checkpoints/cttp/synth-u
curl -L https://huggingface.co/mldi-lab/ConTSG-Bench-Checkpoints/resolve/main/resources/cttp/synth-u/model_configs.yaml \
  -o checkpoints/cttp/synth-u/model_configs.yaml
curl -L https://huggingface.co/mldi-lab/ConTSG-Bench-Checkpoints/resolve/main/resources/cttp/synth-u/clip_model_best.pth \
  -o checkpoints/cttp/synth-u/clip_model_best.pth

# point the released runtime config to your local LongCLIP directory
LONGCLIP_ROOT="$(pwd)/checkpoints/longclip"
sed -i "s|\${LONGCLIP_ROOT}|${LONGCLIP_ROOT}|g" checkpoints/cttp/synth-u/model_configs.yaml
```

### 4. VerbalTS base checkpoint

Download the Pure VerbalTS checkpoint from the official release
(https://huggingface.co/mldi-lab/ConTSG-Bench-Checkpoints) and place it at:

```text
artifacts/no_sae_pure_verbalts/verbalts.ckpt
```

This is the only checkpoint not versioned in this repository (GitHub size
limits). All SAE / classifier weights required for steering are already
included under `results/sae_retrain/` (see below).

## Retained SAE & classifier weights (in-repo)

| Artifact | Path |
| --- | --- |
| Top-K SAE (t = 5–45, Top-K = 48) | `results/sae_retrain/models/t5_45/best.pt` (+ `config.json`, `metrics.json`) |
| Shape classifier (3-stage × 4-shape, on SAE latents) | `results/sae_retrain/classwise_classifiers/t5_45/best.pt` |
| Held-out segment CNN evaluator | `results/sae_retrain/classwise_steering/evaluation_classifier/segment_cnn.pth` |

## SAE training

```bash
# 1. collect residual-layer-1 activations of Pure VerbalTS on synth-u
python -m sae.collect_activations --checkpoint artifacts/no_sae_pure_verbalts/verbalts.ckpt \
  --dataset synth-u --t-start 5 --t-end 45 --output-dir results/sae_retrain/activations

# 2. train the Top-K SAE (fixed t=5–45 recipe)
bash sae/training/train_t5_45.sh

# 3. train the position-aware segment-shape classifier on SAE latents
python -m sae.train_latent_classifier
```

## Steering (classifier guidance on SAE latents)

Two ready-made evaluation entry points:

```bash
# dense full-dim guidance (hybrid13, gamma=6 eta=2560)
python -m sae.evaluate_compositional_full \
  --mode guidance --windows t5_45 --strengths 2560 \
  --guidance-adaptive 6.0 --guidance-full-strength "double peaks,sag" \
  --output-dir results/compositional_guidance_hybrid13 --device cuda

# sparse per-token top-32 guidance (sparse_dynamic_k32_eta5120)
python -m sae.evaluate_control_variants \
  --mode sparse \
  --output-dir results/control_ablations/sparse --device cuda
```

The guidance operator lives in
[`sae/wrappers.py`](sae/wrappers.py) (`LatentClassifierGuidanceWrapper`):
`z ← relu(z − clamp(η · ∇_z mean_stage(−log p_target)))` with σ-relative step
capping, confidence-weighted (`1 − p)^γ` shape scaling, and optional per-token
top-k sparsification (`topk_mode="dynamic"`).

## Repository layout

```text
contsg/          # ConTSG-Bench core: registry, data modules, VerbalTS, eval
                 #   (training / evaluation / visualization pipeline)
sae/             # Top-K SAE model wrapper, training, latent guidance wrappers
configs/         # synth-u training (VerbalTS) and CTTP evaluation configs
results/         # retained weights + two conclusion documents
pyproject.toml   # package metadata and dependencies
```

## Citation

This project builds on ConTSG-Bench (seqml/ConTSG-Bench) and LongCLIP
(zer0int/LongCLIP-GmP-ViT-L-14). Please cite the corresponding works.
