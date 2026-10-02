"""Full main-path repro for diffusets eval: attrs -> indices -> loop batches."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sae.activations import load_generator  # noqa: E402
from sae.eval_any_morph_steering import evaluation_indices, generate_variant_any  # noqa: E402

CKPT = Path("/public/home/liym2024/Verbalts_SAE/results/etth2_2class/seed1/diffusets/20261002_004238_etth2-2class_diffusets/checkpoints/finetune/finetune-epoch=221-val/loss=0.1229.ckpt")
DATA = Path("/public/home/liym2024/Verbalts_SAE/datasets/etth2_2class/etth2_2class_dataset")

device = torch.device("cuda")
model, name = load_generator(CKPT, device)
attrs = np.load(DATA / "test_attrs_idx.npy")
print(f"attrs len={len(attrs)}", flush=True)
indices = evaluation_indices(len(attrs), 0, 0)
print(f"indices len={len(indices)}", flush=True)
emb_all = np.load(DATA / "test_cap_emb.npy")
print(f"cap_emb shape={emb_all.shape}", flush=True)
embeddings = emb_all[indices].astype(np.float32)
print(f"embeddings shape={embeddings.shape}", flush=True)
parts = []
for start in range(0, len(indices), 16):
    end = min(start + 16, len(indices))
    cond = torch.from_numpy(embeddings[start:end]).to(device)
    print(f"iter start={start} cond={tuple(cond.shape)}", flush=True)
    curves = generate_variant_any(model, name, cond, None, 42 + start, transform_block_index=2)
    print(f"  curves={curves.shape}", flush=True)
    parts.append(curves)
    if start >= 32:
        break
print(f"final concat(3 batches) -> {np.concatenate(parts).shape}", flush=True)
