"""Local validation search with roughness guardrails and held-out confirmation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts.iterate_stabilized_steering import ROOT, BASE, CKPT, write


REFERENCE = dict(name="reference", topk=32, eta=5120, gamma=0,
                 guidance_t_range=[5, 45], max_step=1., rel_cap=2.,
                 selection_score="applied", preserve_residual=True)
SPLITS = dict(screen=(768, 128), refine=(896, 256), confirm=(1152, 768))
MSE_MEAN_TOLERANCE = .01
MSE_SEED_TOLERANCE = .02
SHORTLIST = 10
CAUTION = "Validation only; seeds share samples and are not independent datasets"
EXTRA_BASELINES = []
PRIMARY_METRIC = "segment_accuracy"
REFINE_SEEDS = (42, 123, 2026)
CONFIRM_SEEDS = (42, 123, 2026, 314, 2718)


def candidates():
    changes = []
    for k in [8, 16, 24, 32, 48, 64, 96, 128]:
        for cap in [1., 2., 3., 4.]:
            changes.append(dict(topk=k, rel_cap=cap, max_step=cap / 2))
    for window in [[5, 25], [5, 35], [10, 40], [10, 45], [15, 45], [20, 45]]:
        for k in [16, 24, 32, 48, 64]:
            changes.append(dict(topk=k, guidance_t_range=window))
    for eta in [1280, 2560, 10240, 20480]:
        for gamma in [0, 1, 3]:
            changes.append(dict(eta=eta, gamma=gamma))
    for step in [.25, .5, 1.5, 2.]:
        changes.append(dict(max_step=step))
    for k in [24, 48, 64]:
        for eta in [2560, 10240]:
            for window in [[5, 35], [10, 45]]:
                changes.append(dict(topk=k, eta=eta, guidance_t_range=window))
    seen = {json.dumps({k: v for k, v in REFERENCE.items() if k != "name"}, sort_keys=True)}
    pool = []
    for change in changes:
        config = dict(REFERENCE, **change)
        key = json.dumps({k: v for k, v in config.items() if k != "name"}, sort_keys=True)
        if key not in seen:
            seen.add(key)
            config["name"] = f"local_{len(pool):03d}"
            pool.append(config)
    return pool


def assess(report, name):
    v, r = [report["variants"][key] for key in [name, "reference"]]
    dm = (v["mse"]["overall_mse"] / r["mse"]["overall_mse"] - 1)
    da = v["cnn"][PRIMARY_METRIC] - r["cnn"][PRIMARY_METRIC]
    safe = all(v[k] <= r[k] * 1.02 for k in ["roughness", "curvature"])
    return dict(name=name, relative_mse_delta=dm, accuracy_delta=da,
                safe=safe, score=-dm / .01 + da / .01)


def rank(reports, pool):
    rows = []
    for config in pool:
        values = [assess(r, config["name"]) for r in reports]
        dm = sum(v["relative_mse_delta"] for v in values) / len(values)
        da = sum(v["accuracy_delta"] for v in values) / len(values)
        eligible = (all(v["safe"] and v["relative_mse_delta"] <= MSE_SEED_TOLERANCE
                        for v in values) and dm <= MSE_MEAN_TOLERANCE and da > 0)
        rows.append(dict(config=config, eligible=eligible, relative_mse_delta=dm,
                         accuracy_delta=da, score=sum(v["score"] for v in values) / len(values),
                         per_seed=values))
    return sorted(rows, key=ranking_key)


def ranking_key(row):
    return (not row["eligible"], -row["accuracy_delta"], row["relative_mse_delta"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=7.5)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "protocol.json").exists():
        raise ValueError("Refusing to overwrite an existing experiment")
    deadline = time.monotonic() + args.hours * 3600
    pool = candidates()
    write(out / "protocol.json", dict(reference=REFERENCE, candidates=pool, splits=SPLITS,
          guardrail="roughness and curvature <= 1.02 * reference on every selection seed",
          selection="Accuracy first, positive mean accuracy gain vs reference; MSE mean <= +1%, each seed <= +2%",
          mse_mean_tolerance=MSE_MEAN_TOLERANCE, mse_seed_tolerance=MSE_SEED_TOLERANCE,
          caution=CAUTION, shortlist=SHORTLIST, extra_baselines=EXTRA_BASELINES,
          primary_metric=PRIMARY_METRIC, refine_seeds=REFINE_SEEDS,
          confirm_seeds=CONFIRM_SEEDS))

    def evaluate(label, configs, split, seed):
        offset, n = SPLITS[split]
        config_path = out / f"{label}.json"
        write(config_path, [dict(name="pure"), dict(name="sae"), REFERENCE] + EXTRA_BASELINES + configs)
        cmd = [sys.executable, "-m", "sae.eval_morph_steering", "--data-root",
               str(ROOT / "datasets/electricity_15min_semisynth_morph"),
               "--results-root", str(BASE), "--model-checkpoint", str(CKPT),
               "--output-dir", str(out / label), "--config-file", str(config_path),
               "--n-samples", str(n), "--sample-offset", str(offset),
               "--batch-size", "16", "--seed", str(seed), "--device", "cuda",
               "--dynamic-threshold", "6"]
        with (out / f"{label}.log").open("w") as log:
            with subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT) as process:
                while True:
                    write(out / "status.json", dict(state="running", stage=label,
                          heartbeat=time.strftime("%Y-%m-%d %H:%M:%S"), pid=process.pid))
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        process.kill()
                        process.wait()
                        raise TimeoutError("Search time budget exhausted")
                    try:
                        code = process.wait(timeout=min(30, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        pass
                if code:
                    raise RuntimeError(f"{label} failed with exit code {code}; inspect its log")
        return json.loads((out / label / "summary.json").read_text())

    try:
        screen_rows = []
        for start in range(0, len(pool), 6):
            group = pool[start:start + 6]
            report = evaluate(f"screen_{start:03d}", group, "screen", 42)
            screen_rows.extend(rank([report], group))
            write(out / "screen_leaderboard.json", screen_rows)
        selected = [r["config"] for r in sorted(screen_rows, key=ranking_key)
                    if r["eligible"]][:SHORTLIST]
        write(out / "refine_configs.json", selected)
        if selected:
            reports = [evaluate(f"refine_seed{s}", selected, "refine", s) for s in REFINE_SEEDS]
            ranked = rank(reports, selected)
            write(out / "refine_leaderboard.json", ranked)
            winner = [r["config"] for r in ranked if r["eligible"]][:1]
        else:
            winner = []
        write(out / "frozen_final.json", [REFERENCE] + winner)
        confirmations = []
        for seed in CONFIRM_SEEDS:
            report = evaluate(f"confirm_seed{seed}", winner, "confirm", seed)
            confirmations.append(dict(seed=seed, variants=report["variants"]))
            write(out / "confirmation_metrics.json", confirmations)
        if "fresh" in SPLITS:
            fresh = []
            for seed in CONFIRM_SEEDS:
                report = evaluate(f"fresh_seed{seed}", winner, "fresh", seed)
                fresh.append(dict(seed=seed, variants=report["variants"]))
                write(out / "fresh_metrics.json", fresh)
        write(out / "status.json", dict(state="completed", candidate=winner,
              caution=CAUTION + "; selection does not guarantee a confirmed improvement"))
    except Exception as exc:
        write(out / "status.json", dict(state="failed", error=str(exc)))
        raise


if __name__ == "__main__":
    main()