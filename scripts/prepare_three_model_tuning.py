"""Materialize the first, validation-only training LR screen for three models."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT.parent / "ConTSG-Bench"
MODELS = ("bridge", "diffusets", "verbalts")
LEARNING_RATES = (0.0003, 0.001, 0.0001)


def candidate_config(model: str, lr: float, smoke: bool = False) -> dict:
    if model not in MODELS or lr not in LEARNING_RATES:
        raise ValueError("unknown model or learning rate")
    common = yaml.safe_load((ROOT / "configs/generators/verbalts_electricity_v3.yaml").read_text())
    source = (ROOT / "configs/generators/verbalts_electricity_v3.yaml" if model == "verbalts"
              else BENCH / f"configs/generators/{model}_synth-u.yaml")
    cfg = copy.deepcopy(yaml.safe_load(source.read_text()))
    cfg["data"] = common["data"]
    cfg["eval"] = common["eval"]
    cfg["seed"] = 42
    train = cfg["train"]
    train.update(batch_size=32, accumulate_grad_batches=8, num_workers=4,
                 lr=lr, log_grad_norm=False, log_param_norm=False)
    for stage in train["stages"]:
        stage["lr"] = lr
        stage["epochs"] = 60 if stage["name"] == "vae_pretrain" else 80
        stage["early_stopping_patience"] = 20
        if smoke:
            stage["epochs"] = 1
    train["epochs"] = sum(s["epochs"] for s in train["stages"])
    if smoke:
        train.update(batch_size=4, accumulate_grad_batches=1, num_workers=0,
                     limit_train_batches=2, limit_val_batches=2, num_sanity_val_steps=0)
    return cfg


def prepare(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    data = ROOT / "datasets/electricity_15min_semisynth_morph"
    provenance = json.loads((ROOT / "results/electricity_v3/embedding_provenance.json").read_text())
    hashes = {}
    for split in ("train", "valid"):
        caption = data / f"{split}_text_caps.npy"
        digest = hashlib.sha256(caption.read_bytes()).hexdigest()
        if provenance[split]["caption_sha256"] != digest:
            raise ValueError(f"stale {split} caption embeddings")
        hashes[caption.name] = digest
        for path in sorted(data.glob(f"{split}_*.npy")):
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    candidates = []
    for model in MODELS:
        for lr in LEARNING_RATES:
            name = f"{model}_lr{lr:g}"
            for suffix, smoke in (("", False), ("_smoke", True)):
                (output / f"{name}{suffix}.yaml").write_text(
                    yaml.safe_dump(candidate_config(model, lr, smoke), sort_keys=False))
            candidates.append({"name": name, "model": model, "lr": lr})
    manifest = {
        "phase": "pilot_only_not_final_selection", "training_seed": 42,
        "generation_seeds_final": [1, 11, 42], "std_ddof": 1,
        "selection_split": "valid", "test_evaluation": False,
        "candidates": candidates, "input_sha256": hashes,
        "bridge_exemplar": "train[(sample_index * 104729 + 42) % n_train]; all splits",
        "promotion": "Requires validation generation metrics; do not compare loss across models.",
        "notes": "DiffuSETS pilot retains original KL schedule; full training required before judging quality.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args().output.resolve())