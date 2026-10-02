"""Bounded validation search, followed by disjoint confirmation; no test tuning."""
import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/electricity_v3/stabilized_global_soft_1010876"
CKPT = ROOT / "results/electricity_v3/verbalts_global_soft/20260923_145405_electricity_15min_semisynth_morph_verbalts/checkpoints/finetune/finetune-epoch=255-val/loss=0.1281.ckpt"


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def candidates():
    yield dict(name="legacy", topk=32, eta=5120, gamma=6)
    for score, residual, gamma, k in itertools.product(
            ["applied", "gradient"], [True, False], [0, 3, 6], [32, 128]):
        yield dict(name=f"{score}_res{int(residual)}_g{gamma}_k{k}",
                   topk=k, eta=5120, gamma=gamma, selection_score=score,
                   preserve_residual=residual)
    for residual, cap, window in itertools.product(
            [True, False], [.25, 1., 2.], [(5, 45), (5, 20), (20, 45)]):
        yield dict(name=f"dose_res{int(residual)}_c{cap}_t{window[0]}_{window[1]}",
                   topk=32, eta=5120, gamma=0, selection_score="applied",
                   preserve_residual=residual, rel_cap=cap, max_step=.5 * cap,
                   guidance_t_range=window)
    for gamma in [0, 3, 6]:
        yield dict(name=f"dense_g{gamma}", topk=0, eta=5120, gamma=gamma)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=7.5)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.hours * 3600
    rows = []

    def evaluate(label, configs, n, offset, seed, trace=False):
        destination = out / label
        config_path = out / f"{label}.json"
        write(config_path, [dict(name="pure"), dict(name="sae")] + configs)
        command = [sys.executable, "-m", "sae.eval_morph_steering",
                   "--data-root", str(ROOT / "datasets/electricity_15min_semisynth_morph"),
                   "--results-root", str(BASE), "--model-checkpoint", str(CKPT),
                   "--output-dir", str(destination), "--config-file", str(config_path),
                   "--n-samples", str(n), "--sample-offset", str(offset),
                   "--batch-size", "16", "--seed", str(seed), "--device", "cuda",
                   "--dynamic-threshold", "6"]
        if trace:
            command.append("--diagnose")
        write(out / "status.json", dict(stage=label, state="running", command=command,
                                       time=time.strftime("%Y-%m-%d %H:%M:%S")))
        with (out / f"{label}.log").open("w") as log:
            try:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=max(1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                write(out / "status.json", dict(stage=label, state="time_budget_exhausted"))
                return None
        if result.returncode:
            write(out / "status.json", dict(stage=label, state="failed", code=result.returncode))
            return None
        return json.loads((destination / "summary.json").read_text())

    def collect(report, stage):
        pure = report["variants"]["pure"]
        for name, value in report["variants"].items():
            if name in ("pure", "sae"):
                continue
            dm = value["mse"]["overall_mse"] - pure["mse"]["overall_mse"]
            da = value["cnn"]["segment_accuracy"] - pure["cnn"]["segment_accuracy"]
            rows.append(dict(stage=stage, name=name, config=value["config"],
                             mse_delta=dm, accuracy_delta=da, both_improve=dm < 0 and da > 0,
                             mse=value["mse"], cnn=value["cnn"],
                             roughness=value["roughness"], curvature=value["curvature"],
                             truth_roughness=value["truth_roughness"],
                             truth_curvature=value["truth_curvature"]))
        write(out / "leaderboard.json", rows)

    pool = list(candidates())
    for start in range(0, len(pool), 6):
        # Reserve time for independent confirmation instead of tuning forever.
        if deadline - time.monotonic() < 3600:
            break
        stage = f"screen_{start:03d}"
        report = evaluate(stage, pool[start:start + 6], 64, 0, 42, trace=start == 0)
        if report is None:
            return
        collect(report, stage)
    ranked = sorted(rows, key=lambda r: (not r["both_improve"],
                    max(r["mse_delta"], 0) + max(-r["accuracy_delta"], 0),
                    -r["accuracy_delta"], r["mse_delta"]))[:4]
    selected = [r["config"] for r in ranked]
    write(out / "frozen_confirmation_configs.json", selected)
    for seed in [42, 123, 2026]:
        if not selected or time.monotonic() >= deadline:
            break
        stage = f"confirmation_seed{seed}"
        report = evaluate(stage, selected, 128, 128, seed)
        if report is None:
            return
        collect(report, stage)
    write(out / "status.json", dict(state="completed", experiments=len(rows),
          confirmation="Disjoint validation indices 128:256; test set untouched",
          caution="Screening wins are not confirmation; inspect paired CIs and all seeds."))


if __name__ == "__main__":
    main()