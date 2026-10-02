"""Create a non-production 64-combination overfit subset by copying aligned rows."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sae.provenance import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("output directory must be empty")
    attrs = np.load(args.source / "train_attrs_idx.npy")
    combo_ids = attrs[:, 0] * 16 + attrs[:, 1] * 4 + attrs[:, 2]
    indices = np.asarray([np.flatnonzero(combo_ids == i)[0] for i in range(64)])
    args.output.mkdir(parents=True, exist_ok=True)
    aligned = ("ts", "attrs_idx", "text_caps", "cap_emb", "source_indices", "source_scale")
    copied = {}
    for split in ("train", "valid", "test"):
        for name in aligned:
            source = args.source / f"train_{name}.npy"
            if source.exists():
                selected = np.load(source)[indices]
                np.save(args.output / f"{split}_{name}.npy", selected)
                copied[f"{split}_{name}"] = list(selected.shape)
    source_meta = json.loads((args.source / "meta.json").read_text())
    source_meta.update({"name": "electricity_15min_semisynth_morph_overfit64",
                        "purpose": "sanity_check_only_not_for_final_evaluation",
                        "final_split": {"train": 64, "valid": 64, "test": 64},
                        "source_dataset": str(args.source.resolve()),
                        "selected_train_indices": indices.tolist()})
    (args.output / "meta.json").write_text(json.dumps(source_meta, indent=2))
    audit = {"all_64_combinations": len(np.unique(combo_ids[indices])) == 64,
             "source_meta_sha256": sha256_file(args.source / "meta.json"), "copied": copied}
    (args.output / "validation.json").write_text(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()