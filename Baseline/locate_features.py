"""Locate DiffLens-style boost/suppress latent dims with IG attribution.

Steps:

1. Load the frozen VerbalTS checkpoint, the retained Top-K SAE (t5_45) and
   the frozen position-aware latent classifier.
2. Collect residual-layer-1 activations inside the SAE timestep window for
   samples whose caption targets the requested ``(stage, shape)`` class
   (default: Beginning / single peak, i.e. "Begging_peak").
3. Encode the activations with the SAE and run Integrated Gradients
   (baseline = zero latent vector) on the classifier's target-class
   log-probability.
4. Aggregate per-dimension scores, pick the top-k positive dims (boost) and
   top-k negative dims (suppress), and save everything as a JSON artifact
   consumed by ``evaluate_steering.py``.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# Make the repository root importable regardless of the working directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import contsg.models.verbalts  # noqa: F401, E402  (Registry side effect)
from contsg.data.datamodule import TimeSeriesDataset  # noqa: E402
from sae.activations import collect_window, load_verbalts  # noqa: E402
from sae.provenance import (  # noqa: E402
    runtime_metadata,
    sha256_file,
    tensor_state_sha256,
    write_json,
)
from sae.shape_classifier import (  # noqa: E402
    captions_to_shape_ids,
    class_group_key,
    encode_sae_latents,
    load_classifier_checkpoint,
)
from sae.shapes import SHAPE_NAMES, STAGE_NAMES  # noqa: E402
from sae.steering import load_sae_checkpoint  # noqa: E402

from Baseline.ig_attribution import (  # noqa: E402
    aggregate_scores,
    compute_ig_scores,
    locate_topk,
)


def build_parser() -> argparse.ArgumentParser:
    root = _REPO_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-checkpoint",
        type=Path,
        default=root / "artifacts/no_sae_pure_verbalts/verbalts.ckpt",
    )
    parser.add_argument(
        "--sae-checkpoint",
        type=Path,
        default=root / "results/sae_retrain/models/t5_45/best.pt",
    )
    parser.add_argument(
        "--classifier-checkpoint",
        type=Path,
        default=root / "results/sae_retrain/classwise_classifiers/t5_45/best.pt",
    )
    parser.add_argument("--data-root", type=Path, default=root / "datasets/synth-u")
    parser.add_argument("--split", choices=("train", "valid", "test"), default="train")
    parser.add_argument(
        "--target-stage",
        type=int,
        default=0,
        help="Stage index of the located class (0=Beginning, 1=Middle, 2=End)",
    )
    parser.add_argument(
        "--target-shape",
        default="single peak",
        choices=SHAPE_NAMES,
        help="Shape name of the located class (default: single peak at Beginning)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Number of boost dims and of suppress dims to keep (top-k each)",
    )
    parser.add_argument("--ig-steps", type=int, default=32, help="IG integration steps")
    parser.add_argument("--ig-batch-size", type=int, default=64)
    parser.add_argument(
        "--max-class-samples",
        type=int,
        default=2048,
        help="Cap on collected target-class samples for the IG pass",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Collection batch size")
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON artifact (default: results/baseline_difflens_ig/"
        "located_<stage>_<shape>.json)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not 0 <= args.target_stage < len(STAGE_NAMES):
        raise ValueError(f"target-stage must be in [0, {len(STAGE_NAMES) - 1}]")
    if args.topk < 1 or args.ig_steps < 1 or args.ig_batch_size < 1:
        raise ValueError("topk, ig-steps and ig-batch-size must be positive")
    if args.max_class_samples < 1:
        raise ValueError("max-class-samples must be positive")
    shape_id = SHAPE_NAMES.index(args.target_shape)
    group_key = class_group_key(args.target_stage, shape_id)

    repo_root = _REPO_ROOT
    output_path = (
        args.output_json.resolve()
        if args.output_json is not None
        else (
            repo_root
            / "results/baseline_difflens_ig"
            / f"located_{STAGE_NAMES[args.target_stage].lower()}"
            f"_{args.target_shape.replace(' ', '_')}.json"
        ).resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # --- Frozen artifacts ---------------------------------------------------
    sae_path = args.sae_checkpoint.resolve()
    sae, t_range, sae_checkpoint = load_sae_checkpoint(sae_path, device)
    classifier_path = args.classifier_checkpoint.resolve()
    latent_classifier, classifier_meta = load_classifier_checkpoint(
        classifier_path, device
    )
    model_path = args.model_checkpoint.resolve()
    model, model_state_hash = load_verbalts(model_path, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    # --- Target-class sample subset -----------------------------------------
    data_root = args.data_root.resolve()
    captions = np.load(
        data_root / f"{args.split}_text_caps.npy", allow_pickle=True
    ).reshape(-1)
    labels = captions_to_shape_ids(captions)
    target_ids = np.nonzero(labels[:, args.target_stage] == shape_id)[0]
    if len(target_ids) == 0:
        raise ValueError(
            f"split {args.split!r} has no samples targeting {group_key!r}"
        )
    keep = int(min(len(target_ids), args.max_class_samples))
    selected_indices = target_ids[:keep]
    full_dataset = TimeSeriesDataset(data_root, split=args.split, normalize=False)
    subset = Subset(full_dataset, [int(index) for index in selected_indices])
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    print(
        f"collecting {keep} target-class samples ({group_key}) at t in "
        f"{t_range} on split {args.split}"
    )

    # --- Collect h, encode z, attribute --------------------------------------
    activations, t_values, audit = collect_window(
        model, loader, t_range, 1, device, args.seed
    )
    tokens_per_sample = int(audit["tokens_per_sample"])
    if activations.shape[1] != sae.input_dim:
        raise ValueError(
            f"activation dim {activations.shape[1]} != SAE input_dim {sae.input_dim}"
        )
    samples = activations.reshape(-1, tokens_per_sample, activations.shape[-1])
    if samples.shape[1] != latent_classifier.tokens_per_sample:
        raise ValueError(
            f"collected token count {samples.shape[1]} != classifier "
            f"tokens_per_sample {latent_classifier.tokens_per_sample}"
        )
    latents = encode_sae_latents(
        samples, sae, device, args.encode_batch_size
    ).float()
    ig_scores = compute_ig_scores(
        latent_classifier,
        latents,
        args.target_stage,
        shape_id,
        args.ig_steps,
        device,
        args.ig_batch_size,
    )
    scores = aggregate_scores(ig_scores)
    boost_dims, suppress_dims = locate_topk(scores, args.topk)

    payload: dict[str, Any] = {
        "method": "integrated_gradients_multiplicative",
        "ig_baseline": "zero latent vector",
        "steering_formula": (
            "z_new[d] = z[d] * boost_factor for boost dims; "
            "z_new[d] = z[d] * suppress_factor for suppress dims; applied inside "
            "the SAE timestep window to samples whose caption targets the group"
        ),
        "target": {
            "stage": int(args.target_stage),
            "stage_name": STAGE_NAMES[args.target_stage],
            "shape": int(shape_id),
            "shape_name": args.target_shape,
            "group": group_key,
        },
        "topk": args.topk,
        "ig_steps": args.ig_steps,
        "t_range": list(t_range),
        "tokens_per_sample": tokens_per_sample,
        "n_target_class_samples_collected": int(samples.shape[0]),
        "n_target_class_samples_available": int(len(target_ids)),
        "scores": [float(value) for value in scores],
        "score_stats": {
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "min": float(scores.min()),
            "max": float(scores.max()),
        },
        "boost_dims": boost_dims,
        "boost_scores": [float(scores[dim]) for dim in boost_dims],
        "suppress_dims": suppress_dims,
        "suppress_scores": [float(scores[dim]) for dim in suppress_dims],
        "collection_audit": audit,
        "audit": {
            "model_checkpoint": str(model_path),
            "model_checkpoint_sha256": sha256_file(model_path),
            "model_state_sha256": model_state_hash,
            "sae_checkpoint": str(sae_path),
            "sae_checkpoint_sha256": sha256_file(sae_path),
            "sae_state_sha256": sae_checkpoint.get(
                "state_dict_sha256", tensor_state_sha256(sae.state_dict())
            ),
            "classifier_checkpoint": str(classifier_path),
            "classifier_checkpoint_sha256": sha256_file(classifier_path),
            "classifier_state_sha256": tensor_state_sha256(
                latent_classifier.state_dict()
            ),
            "data_root": str(data_root),
            "split": args.split,
            "seed": args.seed,
            "runtime": runtime_metadata(repo_root),
        },
    }
    write_json(output_path, payload)
    print(f"saved located features to {output_path}")
    print(f"boost dims: {boost_dims}")
    print(f"suppress dims: {suppress_dims}")


if __name__ == "__main__":
    main()
