"""Weather36 isolated model configurations and real-data smoke checks."""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/weather36/tuning"
MODELS = ("verbalts", "bridge", "diffusets")


def candidate(model, lr):
    cfg = yaml.safe_load((ROOT / f"configs/generators/{model}_electricity_v3.yaml").read_text())
    cfg["data"].update(name="weather_three_stage_36_morph", seq_length=36,
        data_folder=str(ROOT / "datasets/weather_three_stage_36_morph"))
    cfg["eval"].update(acd_max_lag=12, cache_folder=str(OUTPUT / "cache"), use_cache=False)
    cfg["eval"]["segment_classifier"]["segment_len"] = 12
    cfg["train"].update(batch_size=32, accumulate_grad_batches=8, lr=lr,
        log_grad_norm=False, log_param_norm=False)
    for stage in cfg["train"]["stages"]:
        stage.update(lr=lr, epochs=60 if stage["name"] == "vae_pretrain" else 80,
                     early_stopping_patience=20)
    cfg["train"]["epochs"] = sum(s["epochs"] for s in cfg["train"]["stages"])
    if model == "bridge":
        cfg["model"]["channel_mult"] = [1, 2, 4]
    elif model == "verbalts":
        cfg["model"].update(base_patch=3)
    else:
        cfg["model"].update(num_levels=3, kld_warmup_epochs=40)
    return cfg


def prepare():
    from contsg.config.schema import ExperimentConfig
    OUTPUT.mkdir(parents=True, exist_ok=True)
    candidates = []
    for model in MODELS:
        for lr in (1e-4, 3e-4, 1e-3):
            cfg = candidate(model, lr)
            ExperimentConfig(**cfg)
            name = f"{model}_lr{lr:g}"
            (OUTPUT / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
            candidates.append(name)
    (OUTPUT / "manifest.json").write_text(json.dumps({"candidates": candidates,
        "selection_split": "valid", "test_evaluation": False,
        "status": "PREPARED_NOT_LAUNCHED", "requires": "embedding hashes and all three smoke checks PASS",
        "bridge_sampling": "train-only exemplars; clip_denoised=False (data exceed [-1,1])",
        "diffusets_internal_padding": "36 to 40 inside encoder only; decoder output cropped to 36"}, indent=2))


def smoke(device):
    import torch
    from contsg.config.schema import ExperimentConfig
    from contsg.models.verbalts import VerbalTSModule
    from contsg.models.bridge import BridgeModule
    from contsg.models.diffusets import DiffuSETS
    classes = dict(zip(MODELS, (VerbalTSModule, BridgeModule, DiffuSETS)))
    data = ROOT / "datasets/weather_three_stage_36_morph"
    import hashlib
    provenance = json.loads((data / "embedding_provenance.json").read_text())
    for split in ("train", "valid"):
        assert provenance[split]["caption_sha256"] == hashlib.sha256((data / f"{split}_text_caps.npy").read_bytes()).hexdigest()
    batch = {"ts": torch.tensor(np.load(data / "train_ts.npy")[:2], device=device),
             "cap_emb": torch.tensor(np.load(data / "train_cap_emb.npy")[:2], device=device)}
    batch["tp"] = torch.arange(36, device=device).float().repeat(2, 1)
    batch["bridge_example_ts"] = torch.tensor(np.load(data / "train_ts.npy")[[42, 100]], device=device)
    report = {}
    for name, cls in classes.items():
        torch.manual_seed(42)
        model = cls(ExperimentConfig(**candidate(name, 3e-4))).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        stages = ("vae_pretrain", "finetune") if name == "diffusets" else ("finetune",)
        for stage in stages:
            if name == "diffusets":
                model.set_stage(stage)
            model.train()
            optimizer.zero_grad()
            loss = model(batch)["loss"]
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_batch = dict(batch)
            valid_batch["ts"] = torch.tensor(np.load(data / "valid_ts.npy")[:2], device=device)
            valid_batch["cap_emb"] = torch.tensor(np.load(data / "valid_cap_emb.npy")[:2], device=device)
            assert torch.isfinite(model(valid_batch)["loss"])
            values = model.generate(valid_batch["cap_emb"], n_samples=1,
                bridge_example_ts=batch["bridge_example_ts"], clip_denoised=False)
        assert tuple(values.shape[-2:]) == (36, 1) and values.numel() == 72
        assert torch.isfinite(values).all()
        report[name] = {"status": "PASS", "shape": list(values.shape), "loss": float(loss)}
        (OUTPUT / "smoke_report.json").write_text(json.dumps(report, indent=2))
        print(name, report[name], flush=True)
        del model, optimizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    prepare()
    if args.smoke:
        smoke(args.device)