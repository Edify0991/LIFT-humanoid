from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jp


@dataclass(frozen=True)
class MimicRewardConfig:
    joint_pos_weight: float = 0.45
    joint_vel_weight: float = 0.10
    key_body_weight: float = 0.35
    root_weight: float = 0.10

    joint_pos_sigma: float = 2.0
    joint_vel_sigma: float = 0.1
    key_body_sigma: float = 10.0
    root_sigma: float = 5.0


def _exp_tracking(err_sq: jp.ndarray, sigma: float) -> jp.ndarray:
    return jp.exp(-sigma * err_sq)


def compute_mimic_reward(
    qpos: jp.ndarray,
    qvel: jp.ndarray,
    key_body_pos: jp.ndarray,
    ref_qpos: jp.ndarray,
    ref_qvel: jp.ndarray,
    ref_key_body_pos: jp.ndarray,
    config: MimicRewardConfig,
) -> tuple[jp.ndarray, dict[str, jp.ndarray]]:
    """Computes a DeepMimic-style reward for redirected mocap imitation."""

    joint_pos_err = jp.mean(jp.square(qpos - ref_qpos))
    joint_vel_err = jp.mean(jp.square(qvel - ref_qvel))
    key_body_err = jp.mean(jp.square(key_body_pos - ref_key_body_pos))
    root_err = jp.mean(jp.square(qpos[:7] - ref_qpos[:7]))

    r_joint_pos = _exp_tracking(joint_pos_err, config.joint_pos_sigma)
    r_joint_vel = _exp_tracking(joint_vel_err, config.joint_vel_sigma)
    r_key_body = _exp_tracking(key_body_err, config.key_body_sigma)
    r_root = _exp_tracking(root_err, config.root_sigma)

    total = (
        config.joint_pos_weight * r_joint_pos
        + config.joint_vel_weight * r_joint_vel
        + config.key_body_weight * r_key_body
        + config.root_weight * r_root
    )

    components = {
        "mimic/joint_pos": r_joint_pos,
        "mimic/joint_vel": r_joint_vel,
        "mimic/key_body": r_key_body,
        "mimic/root": r_root,
    }
    return total, components
