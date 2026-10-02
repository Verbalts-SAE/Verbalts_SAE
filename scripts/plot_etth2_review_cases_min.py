import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/public/home/liym2024/Verbalts_SAE")
DATA = ROOT / "datasets/etth2_2class/etth2_2class_dataset"
OUT = ROOT / "outputs/etth2_steering_cases_review"
OUT.mkdir(parents=True, exist_ok=True)

GT_ALL = np.load(DATA / "test_ts.npy")[:, :, 0].astype(float)
LAB_ALL = np.load(DATA / "test_labels.npy")

RUNS = [
    (
        "diffusets_seed1_light",
        "diffusets seed1",
        ROOT / "results/etth2_2class/seed1/diffusets/steering/v2_fulltest_1023308",
        ["pure", "grad_bi_c4_s2"],
        "grad_bi_c4_s2",
    ),
    (
        "verbalts_v2_seed1_boost",
        "verbalts_v2 seed1",
        ROOT / "results/etth2_2class/seed1/verbalts_v2/steering/boost3_fulltest_1023520",
        ["pure", "applied_c24_s7", "applied_c28_s8", "applied_c32_s9"],
        "applied_c28_s8",
    ),
    (
        "bridge_seed42_mild",
        "bridge seed42",
        ROOT / "results/etth2_2class/seed42/bridge/steering/v2_fulltest_1023266",
        ["pure", "applied_c2_s1", "grad_c2_s1"],
        "applied_c2_s1",
    ),
]


def pred(run_dir: Path, name: str) -> np.ndarray:
    data = json.loads((run_dir / f"{name}_predictions.json").read_text())
    return np.array([[int(x.get("peak_count", -9)) for x in row] for row in data])


def marks(pred_row: np.ndarray, target_row: np.ndarray) -> str:
    return " ".join(("✓" if p == t else "×") + f"{p}/{t}" for p, t in zip(pred_row, target_row))


def load_one(tag, model, run_dir, variants, steered):
    summary = json.loads((run_dir / "summary.json").read_text())
    idx = np.asarray(summary["indices"], dtype=int)
    gt = GT_ALL[idx]
    labels = LAB_ALL[idx]
    target = labels + 1
    curves = {v: np.load(run_dir / f"{v}.npy").astype(float) for v in variants}
    preds = {v: pred(run_dir, v) for v in variants}
    mse = {v: ((curves[v] - gt) ** 2).mean(axis=1) for v in variants}
    pure_ok = np.all(preds["pure"] == target, axis=1)
    steer_ok = np.all(preds[steered] == target, axis=1)
    cases = []

    def add(kind, arr, k=2):
        for item in list(arr)[:k]:
            i = int(item)
            if i not in [x[1] for x in cases]:
                cases.append((kind, i))

    add("fixed", np.where((~pure_ok) & steer_ok)[0])
    add("broken", np.where(pure_ok & (~steer_ok))[0])
    add("improved", np.where(((preds[steered] == target).sum(axis=1) > (preds["pure"] == target).sum(axis=1)) & (~steer_ok))[0])
    add("high_mse", np.argsort(-mse[steered]))
    ok_idx = np.where(steer_ok)[0]
    if len(ok_idx):
        add("low_mse_ok", ok_idx[np.argsort(mse[steered][ok_idx])])
    if not cases:
        add("first", range(min(8, len(idx))), k=8)
    return dict(tag=tag, model=model, run_dir=run_dir, variants=variants, steered=steered,
                idx=idx, gt=gt, labels=labels, target=target, curves=curves, preds=preds,
                mse=mse, cases=cases[:8])


def plot_detail(info):
    rows = info["cases"]
    cols = ["GT"] + info["variants"]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(3.0 * len(cols), 2.05 * len(rows)), squeeze=False)
    for r, (kind, i) in enumerate(rows):
        for c, name in enumerate(cols):
            ax = axes[r, c]
            for b in (43, 86):
                ax.axvline(b, color="gray", ls=":", lw=0.6)
            if name == "GT":
                ax.plot(info["gt"][i], color="black", lw=1.4)
                ax.set_title(f"GT {kind}\nrow={i} test={int(info['idx'][i])} lab={info['labels'][i].tolist()}", fontsize=8)
            else:
                ax.plot(info["gt"][i], color="0.78", lw=0.9)
                ax.plot(info["curves"][name][i], lw=1.2)
                ok = bool(np.all(info["preds"][name][i] == info["target"][i]))
                ax.set_title(
                    f"{name} {'OK' if ok else 'WRONG'}\n{marks(info['preds'][name][i], info['target'][i])} mse={info['mse'][name][i]:.2f}",
                    fontsize=7,
                    color="green" if ok else "firebrick",
                )
            ax.set_xticks([0, 43, 86, 127])
            ax.tick_params(labelsize=6)
    fig.suptitle(info["model"] + " representative cases", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = OUT / f"{info['tag']}_cases.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out)


def pick_three(info):
    selected = []
    for want in ["fixed", "high_mse", "low_mse_ok", "broken", "improved"]:
        for kind, i in info["cases"]:
            if kind == want and i not in [x[1] for x in selected]:
                selected.append((kind, i))
                break
        if len(selected) >= 3:
            break
    for kind, i in info["cases"]:
        if len(selected) >= 3:
            break
        if i not in [x[1] for x in selected]:
            selected.append((kind, i))
    return selected[:3]


def plot_combined(infos):
    fig, axes = plt.subplots(len(infos), 3, figsize=(13, 8.5), squeeze=False)
    for r, info in enumerate(infos):
        for c, (kind, i) in enumerate(pick_three(info)):
            ax = axes[r, c]
            st = info["steered"]
            ax.plot(info["gt"][i], "k--", lw=1.3, label="GT")
            ax.plot(info["curves"]["pure"][i], lw=1.1, label="pure")
            ax.plot(info["curves"][st][i], lw=1.1, label=st)
            for b in (43, 86):
                ax.axvline(b, color="gray", ls=":", lw=0.6)
            ax.set_title(
                f"{info['model']} {kind}\nrow={i} test={int(info['idx'][i])} lab={info['labels'][i].tolist()}\n"
                f"pure {marks(info['preds']['pure'][i], info['target'][i])}\n"
                f"steer {marks(info['preds'][st][i], info['target'][i])}\n"
                f"MSE {info['mse']['pure'][i]:.2f}->{info['mse'][st][i]:.2f}",
                fontsize=7,
            )
            ax.set_xticks([0, 43, 86, 127])
            ax.tick_params(labelsize=6)
            if r == 0 and c == 0:
                ax.legend(fontsize=7)
    fig.suptitle("ETTh2 generated case review: GT vs pure vs selected steering", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = OUT / "all_models_representative_cases.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print("saved", out)


def main():
    infos = [load_one(*run) for run in RUNS]
    for info in infos:
        plot_detail(info)
    plot_combined(infos)


if __name__ == "__main__":
    main()