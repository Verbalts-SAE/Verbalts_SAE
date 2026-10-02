"""Reader-health probe for cpu_shape_v1 dynamic steering.

Two search rounds (jobs 1018520 / 1018987, ~37 candidates) produced no
candidate above pure ACCR on cpu_shape_v1, while the transferred
electricity_v3 champion recipe gained +4.65 pp there.  The guidance
gradient is only as informative as the SAE-latent classifier ("reader") on
the inputs it actually sees.  This probe replays the pipeline's single-step
teacher-forcing activation protocol (t ~ U[5, 45], fresh noise, layer-1
residual hook, SAE encode) with generated curves in place of the ground
truth and measures the reader's joint/per-stage shape accuracy on three
curve sources:

  truth    - valid ground-truth curves (reference point, cache ~= 0.974)
  pure     - unsteered generation from the same embeddings
  steered  - generation under the electricity_v3 champion recipe

If the reader stays accurate on generated curves, the bottleneck is the
steering write path; if accuracy collapses there, the guidance target
itself is mis-specified on the generated distribution.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sae.activations import load_verbalts
from sae.provenance import sha256_file, write_json
from sae.shape_classifier import encode_sae_latents, load_classifier_checkpoint
from sae.steering import load_sae_checkpoint

ROOT = Path(__file__).resolve().parents[1]
BATCH = 64
T_RANGE = (5, 45)


def capture_rows(model, batch, device, layer_index=1):
    """One hooked forward pass; returns residual rows (batch * tokens, dim)."""

    captured = []

    def hook(_module, _inputs, output):
        residual = output[0] if isinstance(output, (tuple, list)) else output
        if residual.ndim != 4:
            raise ValueError(f"expected residual (B, C, V, T), got {tuple(residual.shape)}")
        captured.append(
            residual.permute(0, 2, 3, 1).reshape(-1, residual.shape[1]).detach().cpu().float()
        )

    handle = model.verbalts.residual_layers[layer_index].register_forward_hook(hook)
    try:
        with torch.no_grad():
            model({key: value.to(device) if torch.is_tensor(value) else value
                   for key, value in batch.items()})
    finally:
        handle.remove()
    assert len(captured) == 1, "expected exactly one hooked forward pass"
    return captured[0]


def reader_predictions(model, sae, mlp, curves, cap_emb, labels, t_values, device):
    sample_count = len(curves)
    row_blocks = []
    for start in range(0, sample_count, BATCH):
        stop = min(start + BATCH, sample_count)
        size = stop - start
        batch = {
            "ts": torch.as_tensor(curves[start:stop], dtype=torch.float32).reshape(size, -1, 1),
            "tp": torch.arange(curves.shape[1], dtype=torch.float32).repeat(size, 1),
            "cap_emb": torch.as_tensor(cap_emb[start:stop], dtype=torch.float32),
            "t": torch.as_tensor(t_values[start:stop], dtype=torch.long),
        }
        row_blocks.append(capture_rows(model, batch, device))
    rows = torch.cat(row_blocks)
    assert len(rows) % sample_count == 0
    rows_per_sample = len(rows) // sample_count
    assert rows_per_sample == int(mlp.tokens_per_sample), (
        f"token count {rows_per_sample} does not match the classifier {mlp.tokens_per_sample}")
    hidden = rows.reshape(sample_count, rows_per_sample, -1)
    latents = encode_sae_latents(hidden, sae, device, batch_size=256)
    with torch.no_grad():
        logits = mlp(latents.to(device).float())
    predicted = logits.argmax(-1).cpu().numpy()
    matches = predicted == labels
    return dict(
        accr=float(matches.all(1).mean()),
        stage_accuracy=[float(matches[:, stage].mean()) for stage in range(matches.shape[1])],
        predicted=predicted,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--screen-dir", type=Path, required=True,
                   help="screen folder containing pure.npz and the steered npz")
    p.add_argument("--steered", default="ao_e10240")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)

    data = ROOT / "datasets/cpu_shape_v1"
    base = ROOT / "results/cpu_shape_v1/pipeline"
    inputs = json.loads((base / "inputs.json").read_text())
    checkpoint = Path(inputs["checkpoint"])
    assert sha256_file(checkpoint) == inputs["checkpoint_sha256"]
    model, _ = load_verbalts(checkpoint, device)
    sae, _, _ = load_sae_checkpoint(base / "sae/best.pt", device)
    mlp, meta = load_classifier_checkpoint(base / "mlp/best.pt", device)
    assert meta["sae_checkpoint_sha256"] == sha256_file(base / "sae/best.pt")
    for net in (model, sae, mlp):
        net.eval()
        for parameter in net.parameters():
            parameter.requires_grad_(False)

    pure = np.load(args.screen_dir / "pure.npz")
    labels = pure["labels"]
    indices = pure["indices"]
    steered = np.load(args.screen_dir / f"{args.steered}.npz")
    assert np.array_equal(steered["indices"], indices)
    sources = {
        "truth": np.load(data / "valid_ts.npy")[indices].reshape(len(indices), -1),
        "pure": pure["curves"],
        args.steered: steered["curves"],
    }
    cap_emb = np.load(data / "valid_cap_emb.npy")[indices].astype(np.float32)

    report = {}
    for name, curves in sources.items():
        repeats = []
        for repeat in range(args.repeats):
            torch.manual_seed(1234 + repeat)
            t_values = np.random.default_rng(2026 + repeat).integers(
                T_RANGE[0], T_RANGE[1] + 1, size=len(curves))
            result = reader_predictions(
                model, sae, mlp, curves, cap_emb, labels, t_values, device)
            repeats.append(dict(repeat=repeat, torch_seed=1234 + repeat,
                                accr=result["accr"], stage_accuracy=result["stage_accuracy"]))
            print("reader", name, "repeat", repeat, "accr", round(result["accr"], 6),
                  "stages", [round(v, 6) for v in result["stage_accuracy"]], flush=True)
        accrs = [row["accr"] for row in repeats]
        report[name] = dict(accr_mean=float(np.mean(accrs)), accr_std=float(np.std(accrs)),
                            repeats=repeats)
    write_json(out / "reader_health.json", dict(
        screen_dir=str(args.screen_dir), steered=args.steered, repeats=args.repeats,
        batch=BATCH, t_range=list(T_RANGE), curves=len(labels), report=report))
    print(json.dumps({name: dict(accr=round(v["accr_mean"], 4), std=round(v["accr_std"], 4))
                      for name, v in report.items()}, indent=2))


if __name__ == "__main__":
    main()
