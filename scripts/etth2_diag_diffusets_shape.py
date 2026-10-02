"""Quick shape diagnostic for DiffuSETS.generate on etth2_2class."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sae.activations import load_generator  # noqa: E402

CKPT = Path("/public/home/liym2024/Verbalts_SAE/results/etth2_2class/seed1/diffusets/20261002_004238_etth2-2class_diffusets/checkpoints/finetune/finetune-epoch=221-val/loss=0.1229.ckpt")
DATA = Path("/public/home/liym2024/Verbalts_SAE/datasets/etth2_2class/etth2_2class_dataset")

device = torch.device("cuda")
model, name = load_generator(CKPT, device)
print(f"model name={name}, type={type(model).__name__}", flush=True)
emb = np.load(DATA / "test_cap_emb.npy")[:16].astype(np.float32)
cond = torch.from_numpy(emb).to(device)
print(f"cond shape={tuple(cond.shape)}", flush=True)
with torch.no_grad():
    g = model.generate(cond, n_samples=1, sampler="ddim")
print(f"generate -> {tuple(g.shape)}", flush=True)
g2 = model.generate(cond, n_samples=2, sampler="ddim")
print(f"generate(n=2) -> {tuple(g2.shape)}", flush=True)
