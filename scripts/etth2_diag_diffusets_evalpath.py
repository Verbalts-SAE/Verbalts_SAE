"""Reproduce the eval-path generation for diffusets to find the shape bug."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sae.activations import load_generator  # noqa: E402
from sae.eval_any_morph_steering import generate_variant_any  # noqa: E402

CKPT = Path("/public/home/liym2024/Verbalts_SAE/results/etth2_2class/seed1/diffusets/20261002_004238_etth2-2class_diffusets/checkpoints/finetune/finetune-epoch=221-val/loss=0.1229.ckpt")
DATA = Path("/public/home/liym2024/Verbalts_SAE/datasets/etth2_2class/etth2_2class_dataset")

device = torch.device("cuda")
model, name = load_generator(CKPT, device)
print(f"model name={name}", flush=True)
emb = np.load(DATA / "test_cap_emb.npy")[:16].astype(np.float32)
cond = torch.from_numpy(emb).to(device)
curves = generate_variant_any(model, name, cond, None, 42, transform_block_index=2)
print(f"generate_variant_any -> {curves.shape}", flush=True)
# also mimic the loop slices
parts = []
for start in range(0, 32, 16):
    c = generate_variant_any(model, name, torch.from_numpy(emb[start:start+16]).to(device), None, 42+start, transform_block_index=2)
    print(f"batch {start}: {c.shape}", flush=True)
    parts.append(c)
print(f"concat -> {np.concatenate(parts).shape}", flush=True)
