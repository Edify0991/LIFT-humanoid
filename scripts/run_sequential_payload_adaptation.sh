#!/usr/bin/env bash
set -euo pipefail

# Minimal sequential adaptation example:
# phase1: left-load -> phase2: right-load -> phase3: offset-load (via inject delay)

BASE_CKPT=${1:-models/G1LowDimJoystickFlatTerrain_policyXXXX.pkl}
WM_CKPT=${2:-logs/G1LowDimPayloadWalking-YYYYmmdd-HHMMSS/wm_states/wm_stateXX.pkl}

# 1) left-load (no distillation)
CUDA_VISIBLE_DEVICES=0 python finetune.py \
  --env_name=G1LowDimPayloadWalking \
  --suffix=seq_left \
  --ac_training_state_path="$BASE_CKPT" \
  --wm_training_state_path="$WM_CKPT"

# 2) right-load with distillation ON (set in lift_configs.py wm_config.distillation_coef > 0)
CUDA_VISIBLE_DEVICES=0 python finetune.py \
  --env_name=G1LowDimPayloadWalking \
  --suffix=seq_right_distill \
  --ac_training_state_path="$BASE_CKPT" \
  --wm_training_state_path="$WM_CKPT"

# 3) delayed injection setting
CUDA_VISIBLE_DEVICES=0 python finetune.py \
  --env_name=G1LowDimPayloadWalking \
  --suffix=seq_delay_distill \
  --ac_training_state_path="$BASE_CKPT" \
  --wm_training_state_path="$WM_CKPT"
