"""Generate the verbalts steering grid config (central-composite style).

Axes (each sweeps one dimension around the previous 12-variant center):
  eta        @ topk=32 gamma=6 : 640..20480 (12 values, dense around 1280-8192)
  topk       @ eta=5120 gamma=6: 2,4,8,16,32,48,64,96,128
  gamma      @ eta=5120 topk=32: 0,1,2,3,4,6,9
  rel_cap    @ eta=5120 topk=32 gamma=6: 0.25,0.5,1.0,2.0,4.0
  t_window   @ eta=5120 topk=32 gamma=6: (5,20),(30,45)
  head_loc   @ eta=5120 topk=32 gamma=6: head 0,1,2

Output: a JSON list accepted by eval_morph_steering --config-file
(always starts with pure + sae).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CENTRE = dict(eta=5120, gamma=6, topk=32)


def build() -> list[dict]:
    variants: list[dict] = [dict(name="pure"), dict(name="sae")]

    for eta in (640, 1280, 1920, 2560, 3200, 3840, 4480, 5120, 6400, 8192, 10240, 20480):
        variants.append(dict(name=f"eta{eta}", eta=eta, **{k: CENTRE[k] for k in ("gamma", "topk")}))
    for topk in (2, 4, 8, 16, 32, 48, 64, 96, 128):
        variants.append(dict(name=f"k{topk}", topk=topk, **{k: CENTRE[k] for k in ("eta", "gamma")}))
    for gamma in (0, 1, 2, 3, 4, 6, 9):
        variants.append(dict(name=f"g{gamma}", gamma=gamma, **{k: CENTRE[k] for k in ("eta", "topk")}))
    for rel_cap in (0.25, 0.5, 1.0, 2.0, 4.0):
        variants.append(dict(name=f"cap{rel_cap:g}", rel_cap=rel_cap, **CENTRE))
    for tmin, tmax in ((5, 20), (30, 45)):
        variants.append(dict(name=f"win{tmin}_{tmax}", guidance_t_range=[tmin, tmax], **CENTRE))
    for head in (0, 1, 2):
        variants.append(dict(name=f"head{head}", objective_head_indices=[head], **CENTRE))

    names = [v["name"] for v in variants]
    assert len(names) == len(set(names)), "duplicate variant names"
    assert variants[:2] == [dict(name="pure"), dict(name="sae")], "pure/sae prefix broken"
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    variants = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(variants) + "\n")
    print(f"wrote {len(variants)} variants to {args.output}")


if __name__ == "__main__":
    main()
