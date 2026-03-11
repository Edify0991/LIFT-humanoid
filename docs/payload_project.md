# Payload Proprioceptive Adaptation (Reproducible MVP)

> Scope: **MVP first** = Task A (Asymmetric-Load Walking) + baselines **No-Adapt / DR-RL / Ours**.

## Step 0 — 仓库入口与观测约束
- Policy pretrain入口：`train_in_mujoco_playground.py`
- WM pretrain入口：`train_wm_from_file.py`
- 安全微调入口：`finetune.py`
- Payload 环境：`brax_env/brax/envs/g1_lowdim_payload_walking.py`

观测约束：
- actor 只用 `state`。
- payload 真值仅拼接到 `privileged_state`（用于 critic/WM），不进入 actor。

## Step 1 — Payload 注入机制（Task A）
环境 `G1LowDimPayloadWalking` 支持：
- train-time 质量随机：`payload_mass_range=(m_min,m_max)`
- eval-time 固定质量：`eval_payload_mass`
- 左/右侧负载：`payload_side`
- 注入时刻：`payload_inject_step`

最小 smoke（本地安装 brax 后）：
```bash
PYTHONPATH=brax_env python -c "from brax.envs.g1_lowdim_payload_walking import G1LowDimPayloadWalking; print('ok')"
```

## Step 2 — 训练入口兼容
### 2.1 预训练（复用已有 walking policy）
```bash
CUDA_VISIBLE_DEVICES=0 python train_in_mujoco_playground.py \
  --env_name=G1LowDimJoystickFlatTerrain \
  --domain_randomization --num_timesteps 40000000 --save_buffer_data
```
输出：`logs/<exp>/policies/*.pkl`, `logs/<exp>/buffer_data/*`

### 2.2 WM 预训练（payload env）
```bash
CUDA_VISIBLE_DEVICES=0 python train_wm_from_file.py \
  --env_name=G1LowDimPayloadWalking \
  --data_path logs/G1LowDimJoystickFlatTerrain-YYYYmmdd-HHMMSS
```
输出：`logs/<exp>/wm_states/wm_state*.pkl`

### 2.3 基于 WM 的安全微调（deterministic in env）
```bash
CUDA_VISIBLE_DEVICES=0 python finetune.py \
  --env_name=G1LowDimPayloadWalking \
  --suffix=payload \
  --ac_training_state_path=models/G1LowDimJoystickFlatTerrain_policyXXXX.pkl \
  --wm_training_state_path=logs/G1LowDimPayloadWalking-YYYYmmdd-HHMMSS/wm_states/wm_stateXX.pkl
```
输出：`logs/<exp>/policy_pkl/*.pkl`, `logs/<exp>/wm_pkl/*.pkl`

## Step 3 — 交互潜变量 ψ 与条件 WM
已接入：
- `world_model/interaction_latent.py`：ψ定义 + heuristic fallback。
- WM输入条件化：`[wm_obs, action, z_t]`（通过 `interaction_latent_dim` 开关）。

关键配置：`lift_configs.py`
- `pretrain_wm_config(...).interaction_latent_dim`
- `finetune_sac_config(...).world_model_config.interaction_latent_dim`

## Step 4 — 评测与基线
### 4.1 单基线 sweep
```bash
PYTHONPATH=. python scripts/eval_payload_sweep.py \
  --env_name G1LowDimPayloadWalking \
  --policy_ckpt <CKPT_PATH> \
  --baseline_name Ours \
  --masses 0,2,4,6,8,10 \
  --num_episodes 10 \
  --seed 0 \
  --out_dir logs/payload_eval
```
输出：
- `payload_sweep_<baseline>.csv`
- `success_vs_mass_<baseline>.png`
- `traj_<baseline>_m*_ep*.npz`

### 4.2 基线定义（MVP）
- **No-Adapt**：直接评估 pretrain `π_prior`
- **DR-RL**：在 payload env 训练但不启用 latent（`interaction_latent_dim=0`）
- **Ours**：WM + latent + 安全微调

分别运行上面的 sweep（`--baseline_name` 改为 `NoAdapt/DRRL/Ours`），然后合并：
```bash
python scripts/compare_payload_baselines.py \
  --csvs logs/payload_eval/payload_sweep_NoAdapt.csv,logs/payload_eval/payload_sweep_DRRL.csv,logs/payload_eval/payload_sweep_Ours.csv \
  --out_dir logs/payload_eval
```
输出：`payload_baselines_merged.csv`, `payload_baselines_success.png`

## Step 5 — 轨迹差异分析
```bash
python scripts/plot_joint_diff.py --traj_dir logs/payload_eval --baseline_mass 0.0
```
输出：`joint_diff.csv`, `joint_diff.png`

---

## 备注
- 本版本先完成 Task A。Task B（STS）与自蒸馏顺序适应脚本可在此基础上扩展。
- 日志结构保持与仓库现有 `logs/<exp>/...` 兼容。
