# Payload Proprioceptive Adaptation MVP

## 0) 现有三阶段入口
- Policy pretrain: `train_in_mujoco_playground.py`
- WM pretrain: `train_wm_from_file.py`
- Model-based finetune: `finetune.py`

本原型新增 Brax 环境：`G1LowDimPayloadWalking`（`brax_env/brax/envs/g1_lowdim_payload_walking.py`）。

## 1) 预训练（可复用已有）
可直接复用仓库已有 G1 checkpoint；若重训可先跑：

```bash
CUDA_VISIBLE_DEVICES=0 python train_in_mujoco_playground.py \
  --env_name=G1LowDimJoystickFlatTerrain \
  --domain_randomization --num_timesteps 40000000 --save_buffer_data
```

## 2) Payload-WM 预训练
要求 `--data_path` 指向包含 `buffer_data/` 的日志目录。

```bash
CUDA_VISIBLE_DEVICES=0 python train_wm_from_file.py \
  --env_name=G1LowDimPayloadWalking \
  --data_path logs/G1LowDimJoystickFlatTerrain-YYYYmmdd-HHMMSS
```

## 3) Payload 上基于 WM 安全微调
```bash
CUDA_VISIBLE_DEVICES=0 python finetune.py \
  --env_name=G1LowDimPayloadWalking \
  --suffix=payload \
  --ac_training_state_path=models/G1LowDimJoystickFlatTerrain_policyXXXX.pkl \
  --wm_training_state_path=logs/G1LowDimPayloadWalking-YYYYmmdd-HHMMSS/wm_states/wm_stateXX.pkl
```

## 4) 评测：success-vs-mass 扫描 + 轨迹导出
```bash
python scripts/eval_payload_sweep.py \
  --policy_ckpt models/G1LowDimJoystickFlatTerrain_policyXXXX.pkl \
  --masses 0,2,4,6,8,10 \
  --num_episodes 5 \
  --out_dir logs/payload_eval
```

输出：
- `payload_sweep.csv`
- `success_vs_mass.png`
- `traj_m*_ep*.npz`（q/qd/action）

## 5) 轨迹差异分析
```bash
python scripts/plot_joint_diff.py --traj_dir logs/payload_eval --baseline_mass 0.0
```

输出：
- `joint_diff.csv`
- `joint_diff.png`

## 实现说明（MVP）
- policy 输入保持 `state`（本体感觉）；不输入 payload 真值。
- payload 真值仅拼接到 `privileged_state` 尾部，用于 critic/WM。
- `world_model/interaction_latent.py` 提供 ψ 编码器定义与启发式 `z_t`；WM 已支持条件输入 `concat([wm_obs, action, z_t])`。
- 安全微调仍沿用 LIFT 的 deterministic-in-env 与 model rollout 探索策略。
