# LIFT 三阶段框架代码映射 + Mimic 扩展说明

## 1) Policy Pretraining（大规模仿真预训练）

入口脚本：`train_in_mujoco_playground.py`

核心流程：
1. 读取环境默认配置：`registry.get_default_config(env_name)`。
2. 读取 SAC 配置：`pretrain_sac_config(env_name)`。
3. 构建网络：`sac_networks.make_sac_networks(...)`。
4. 启动训练：`policy_pretrain.train.train(...)`。

关键函数：
- `policy_pretrain.train.train`：SAC 主训练循环，负责
  - 环境并行采样；
  - replay buffer 存取；
  - actor/critic/alpha 更新；
  - 定期评估与 checkpoint 导出。
- `progress(...)`：训练中的日志、渲染、策略参数保存回调。

## 2) World Model Pretraining（物理先验世界模型预训练）

入口脚本：`train_wm_from_file.py`

核心流程：
1. 从 SAC 预训练日志目录中读取 `buffer_data/`。
2. 根据环境选择 Brax 侧模型环境与机器人配置。
3. 调用 `world_model.pretrain_wm.train(...)` 进行 WM 训练。

关键函数：
- `world_model.pretrain_wm.train`：负责世界模型训练循环、验证集评估与参数持久化。
- `add_kwargs_to_fn`：将配置动态注入环境构造函数。

## 3) Finetuning（真实/新域高效微调）

入口脚本：`finetune.py`

核心流程：
1. 读取 `finetune_sac_config(env_name)`。
2. 构建真实环境与 ModelEnv。
3. 构建 actor-critic 网络（`sac_networks`）与 world model 网络（`wm_networks`）。
4. 调用 `world_model.finetune_wm_ac.train(...)` 联合微调。

关键函数：
- `wm_networks.make_inference_fn`：将世界模型参数封装为可 rollout 的动力学/奖励预测器。
- `ar_rollout_scan`：用于分析 AR rollout 的模型误差累积。
- `train_progress(...)`（`finetune.py` 内部）：记录奖励、保存最优策略、可视化模型误差。

---

## Mimic 任务扩展（重定向人体动捕 + SAC）

本仓库新增了 Mimic 扩展基础模块：

- `mimic/reference_motion.py`
  - `ReferenceClip`：封装单条 mocap 片段（`qpos/qvel/key_body_pos/dt`）。
  - `load_reference_clip(path)`：从 `npz` 载入重定向动作。
  - `MotionLibrary`：多动作片段采样与循环 frame 对齐。

- `mimic/rewards.py`
  - `MimicRewardConfig`：模仿奖励权重与核参数。
  - `compute_mimic_reward(...)`：计算 joint pose/vel、关键点、root 四项 imitation reward。

- `lift_configs.py`
  - `mimic_sac_config()`：Mimic 任务推荐 SAC 超参数模板。
  - `mimic_reward_config()`：Mimic 奖励模板。

### 推荐接入方式

1. 在目标 humanoid 环境（MuJoCo Playground 或 Brax）新增 mimic 版本环境：
   - reset 时采样 clip 与起始 frame；
   - 每步根据 `env_step` 取 reference frame；
   - 在 reward 中调用 `compute_mimic_reward(...)`；
   - 终止条件增加大姿态偏差早停。
2. 观测中加入 phase / ref latent / 若干关键参考量。
3. 先用 `mimic_sac_config()` 做大规模预训练，再按 LIFT 原流程训练 WM 与微调。

