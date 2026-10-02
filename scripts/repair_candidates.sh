#!/usr/bin/env bash
# Repair candidate evaluation:
#   - bridge/diffusets: generation already succeeded, only CTTP scoring was missing.
#   - verbalts: full re-run (config file fixed to pure+sae prefix).
set -euo pipefail
INDEX=${1:?pass candidate index 0-8}
ROOT=/public/home/liym2024/Verbalts_SAE
PY=/public/home/liym2024/.conda/envs/my_torch/bin/python
DATA=$ROOT/datasets/electricity_15min_semisynth_morph
EVALROOT=$ROOT/results/electricity_v3/candidate_eval
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

MODELS=(bridge bridge bridge diffusets diffusets diffusets verbalts verbalts verbalts)
LRS=(0.0001 0.0003 0.001 0.0001 0.0003 0.001 0.0001 0.0003 0.001)
MODEL=${MODELS[$INDEX]}
LR=${LRS[$INDEX]}
NAME=${MODEL}_lr${LR}
OUT=$EVALROOT/runs/$NAME

if [ "$MODEL" = "verbalts" ]; then
    # gen already produced by the repaired run; only CTTP scoring is missing.
    if [ ! -f "$OUT/gen/pure.npy" ] || [ ! -f "$OUT/gen/summary.json" ]; then
        echo "MISSING_GEN $NAME"
        exit 3
    fi
    if [ ! -f "$OUT/indices.json" ]; then
        cd "$ROOT"
        PYTHONPATH="$ROOT" "$PY" - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
report = json.loads((out / "gen/summary.json").read_text())
(out / "indices.json").write_text(json.dumps(report["indices"]))
PY
    fi
    cd "$ROOT"
    PYTHONPATH="$ROOT" "$PY" -m scripts.score_curves_cttp \
        --curves "$OUT/gen/pure.npy" --captions "$DATA/valid_text_caps.npy" \
        --indices-json "$OUT/indices.json" --device cuda --output "$OUT/cttp.json"
    echo REPAIRED > "$OUT/status.txt"
    echo "CTTP_REPAIRED $NAME"
    exit 0
fi

if [ ! -f "$OUT/gen/curves.npy" ] || [ ! -f "$OUT/gen/summary.json" ]; then
    echo "MISSING_GEN $NAME"
    exit 3
fi
cd "$ROOT"
PYTHONPATH="$ROOT" "$PY" -m scripts.score_curves_cttp \
    --curves "$OUT/gen/curves.npy" --captions "$DATA/valid_text_caps.npy" \
    --caption-limit 512 --device cuda --output "$OUT/cttp.json"
echo REPAIRED > "$OUT/status.txt"
echo "CTTP_REPAIRED $NAME"
