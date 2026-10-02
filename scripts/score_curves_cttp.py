"""Score saved generated curves with the electricity_v3 CTTP model.

CTTP = mean over samples of <E_ts(curve_i), E_text(caption_i)>, the diagonal
text-time-series alignment.  Higher is better; steering must keep it stable
relative to the pure baseline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/cttp/cttp_electricity_v3.yaml"
DEFAULT_CKPT = (
    ROOT / "results/electricity_v3/cttp_1011230/full"
    / "20260924_163043_electricity_15min_semisynth_morph_cttp"
    / "checkpoints/finetune/last.ckpt"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curves", type=Path, required=True, help="(N, 128, 1) generated curves")
    parser.add_argument("--captions", type=Path, required=True, help="(N, 1) text captions npy")
    parser.add_argument("--caption-offset", type=int, default=0,
                        help="start row of the split used for generation")
    parser.add_argument("--caption-limit", type=int, default=None,
                        help="number of rows generated; None reads the whole file")
    parser.add_argument("--indices-json", type=Path, default=None,
                        help="JSON list of split row indices matching the curves 1:1 "
                             "(overrides caption-offset/limit)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"CTTP checkpoint not found: {args.checkpoint}")
    device = torch.device(args.device)
    from contsg.config.schema import ExperimentConfig
    from contsg.models.cttp import CTTPModel

    # Verbalts_SAE configs carry dataset-only extras that the ConTSG-Bench
    # schema rejects (pydantic extra_forbidden).  Strip them before building
    # the ExperimentConfig object used for model loading.
    raw = yaml.safe_load(args.config.read_text())
    if isinstance(raw, dict):
        data = raw.get("data")
        if isinstance(data, dict):
            for key in ("caption_variant",):
                data.pop(key, None)
        cond = raw.get("condition")
        if isinstance(cond, dict):
            text = cond.get("text")
            if isinstance(text, dict):
                for key in ("embedding_key", "text_projector", "num_stages"):
                    text.pop(key, None)
    config = ExperimentConfig(**raw)
    config.device = str(device)
    model = CTTPModel.load_from_checkpoint(str(args.checkpoint), config=config, strict=False)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    curves = np.load(args.curves)
    if curves.ndim == 2 and curves.shape[1] == 128:
        curves = curves[:, :, None]
    if curves.ndim != 3 or curves.shape[1:] != (128, 1):
        raise ValueError(f"curves must have shape (N, 128, 1), got {curves.shape}")
    captions = np.load(args.captions, allow_pickle=True).reshape(-1)
    if args.indices_json is not None:
        indices = json.loads(args.indices_json.read_text())
        if not isinstance(indices, list) or any(not isinstance(i, int) for i in indices):
            raise ValueError("indices-json must contain a list of integers")
        captions = captions[np.asarray(indices, dtype=np.int64)]
    else:
        stop = (args.caption_offset + args.caption_limit) if args.caption_limit else None
        captions = captions[args.caption_offset:stop]
    if len(captions) != len(curves):
        raise ValueError(f"caption count {len(captions)} != curve count {len(curves)}")
    captions = [str(c) for c in captions]

    values: list[float] = []
    with torch.no_grad():
        for start in range(0, len(curves), args.batch_size):
            stop = min(start + args.batch_size, len(curves))
            ts = torch.from_numpy(np.array(curves[start:stop], copy=True)).to(
                device=device, dtype=torch.float32)
            text = captions[start:stop]
            ts_emb = model.get_ts_embedding(ts)      # (B, D)
            text_emb = model.get_text_embedding(text)  # (B, D)
            sims = (ts_emb * text_emb).sum(dim=-1)   # diagonal alignment
            values.extend(sims.detach().cpu().tolist())
    report = {
        "cttp_mean": float(np.mean(values)),
        "cttp_std": float(np.std(values)),
        "cttp_per_sample": values,
        "n_samples": len(values),
        "config": str(args.config),
        "config_sha256": sha256_file(args.config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "curves": str(args.curves),
        "curves_sha256": sha256_file(args.curves),
        "captions": str(args.captions),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cttp_per_sample"}),
          flush=True)


if __name__ == "__main__":
    main()
