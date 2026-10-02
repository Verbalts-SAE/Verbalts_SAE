#!/usr/bin/env bash
# Evaluate one three-model candidate (index 0-8) on the validation split.
set -euo pipefail
INDEX=${1:?pass candidate index 0-8}
ROOT=/public/home/liym2024/Verbalts_SAE
BENCH=/public/home/liym2024/ConTSG-Bench
PY=/public/home/liym2024/.conda/envs/my_torch/bin/python
DATA=$ROOT/datasets/electricity_15min_semisynth_morph
CFG=$ROOT/results/electricity_v3/candidate_eval/configs
EVALROOT=$ROOT/results/electricity_v3/candidate_eval
CNN=$ROOT/results/electricity_v3/cnn/segment_cnn.pth
BANK=$ROOT/results/electricity_v3/steering_infra/bridge_exemplar_bank.pt
VROOT=$ROOT/results/electricity_v3/three_model_tuning_v1/runs
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

MODELS=(bridge bridge bridge diffusets diffusets diffusets verbalts verbalts verbalts)
LRS=(0.0001 0.0003 0.001 0.0001 0.0003 0.001 0.0001 0.0003 0.001)
MODEL=${MODELS[$INDEX]}
LR=${LRS[$INDEX]}
NAME=${MODEL}_lr${LR}
OUT=$EVALROOT/runs/$NAME
mkdir -p "$OUT"
echo "HOST=$(hostname) INDEX=$INDEX MODEL=$MODEL LR=$LR" > "$OUT/status.txt"

PILOT=$(ls -d "$VROOT/$NAME/pilot"/*/ | head -1)
CKPT="$PILOT/checkpoints/finetune/last.ckpt"
echo "PILOT=$PILOT" >> "$OUT/status.txt"
echo "CKPT=$CKPT" >> "$OUT/status.txt"
trap 'echo FAILED >> "$OUT/status.txt"' ERR

if [ "$MODEL" = "bridge" ]; then
    cd "$BENCH"
    PYTHONPATH="$BENCH" "$PY" -m sae.evaluate_bridge_steering \
        --config "$CFG/$NAME.yaml" --checkpoint "$CKPT" \
        --data-root "$DATA" --evaluation-classifier "$CNN" \
        --exemplar-bank "$BANK" --mode baseline \
        --split valid --limit 512 --batch-size 128 --seed 42 \
        --device auto --output-dir "$OUT/gen"
    PYTHONPATH="$ROOT" "$PY" -m scripts.score_curves_cttp \
        --curves "$OUT/gen/curves.npy" --captions "$DATA/valid_text_caps.npy" \
        --caption-limit 512 --device cuda --output "$OUT/cttp.json"
elif [ "$MODEL" = "diffusets" ]; then
    cd "$BENCH"
    PYTHONPATH="$BENCH" "$PY" -m sae.evaluate_diffusets_steering \
        --config "$CFG/$NAME.yaml" --checkpoint "$CKPT" \
        --data-root "$DATA" --evaluation-classifier "$CNN" \
        --eta 0 --split valid --limit 512 --batch-size 128 --seed 42 \
        --device auto --output-dir "$OUT/gen"
    PYTHONPATH="$ROOT" "$PY" -m scripts.score_curves_cttp \
        --curves "$OUT/gen/curves.npy" --captions "$DATA/valid_text_caps.npy" \
        --caption-limit 512 --device cuda --output "$OUT/cttp.json"
else
    cd "$ROOT"
    cat > "$OUT/pure.json" <<'EOF'
[{"name": "pure"}, {"name": "sae"}]
EOF
    PYTHONPATH="$ROOT" "$PY" -m sae.eval_morph_steering \
        --data-root "$DATA" --results-root "$ROOT/results/electricity_v3" \
        --model-checkpoint "$CKPT" --output-dir "$OUT/gen" \
        --config-file "$OUT/pure.json" \
        --n-samples 512 --batch-size 16 --seed 42 --device cuda \
        --dynamic-threshold 6
    PYTHONPATH="$ROOT" "$PY" - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
report = json.loads((out / "gen/summary.json").read_text())
(out / "indices.json").write_text(json.dumps(report["indices"]))
PY
    PYTHONPATH="$ROOT" "$PY" -m scripts.score_curves_cttp \
        --curves "$OUT/gen/pure.npy" --captions "$DATA/valid_text_caps.npy" \
        --indices-json "$OUT/indices.json" --device cuda --output "$OUT/cttp.json"
fi
echo COMPLETED > "$OUT/status.txt"
echo "CANDIDATE_EVAL_DONE $NAME"
