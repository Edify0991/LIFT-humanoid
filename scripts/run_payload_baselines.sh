#!/usr/bin/env bash
set -euo pipefail

NO_ADAPT_CKPT=${1:-models/G1_prior.pkl}
DR_RL_CKPT=${2:-models/G1_payload_drrl.pkl}
OURS_CKPT=${3:-models/G1_payload_ours.pkl}
OUT_DIR=${4:-logs/payload_eval}

PYTHONPATH=. python scripts/eval_payload_sweep.py --policy_ckpt "$NO_ADAPT_CKPT" --baseline_name NoAdapt --out_dir "$OUT_DIR"
PYTHONPATH=. python scripts/eval_payload_sweep.py --policy_ckpt "$DR_RL_CKPT" --baseline_name DRRL --out_dir "$OUT_DIR"
PYTHONPATH=. python scripts/eval_payload_sweep.py --policy_ckpt "$OURS_CKPT" --baseline_name Ours --out_dir "$OUT_DIR"

python scripts/compare_payload_baselines.py \
  --csvs "$OUT_DIR/payload_sweep_NoAdapt.csv,$OUT_DIR/payload_sweep_DRRL.csv,$OUT_DIR/payload_sweep_Ours.csv" \
  --out_dir "$OUT_DIR"
