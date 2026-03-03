from __future__ import annotations

from typing import Dict

import jax
from jax import numpy as jp

from brax.envs.base import State
from brax.envs.g1_lowdim_joystick import G1LowDimJoystick
from brax.math import quat_to_eulerzyx
from brax.robots.g1.utils import g1Utils


class G1LowDimPayloadWalking(G1LowDimJoystick):
    """Payload-robust walking prototype with unknown disturbance.

    MVP implementation uses an equivalent generalized disturbance torque pattern
    proportional to payload mass. The policy sees only proprioceptive `state`;
    payload mass is appended only to `privileged_state` for critic/WM training.
    """

    def __init__(
        self,
        payload_mass_range=(0.0, 8.0),
        eval_payload_mass: float | None = None,
        payload_side: str = "left",
        payload_inject_step: int = 0,
        payload_torque_scale: float = 0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._payload_mass_range = payload_mass_range
        self._eval_payload_mass = eval_payload_mass
        self._payload_side = payload_side
        self._payload_inject_step = payload_inject_step
        self._payload_torque_scale = payload_torque_scale
        # equivalent disturbance on hip-roll/hip-yaw for asymmetry
        self._left_disturbance = jp.array([0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self._right_disturbance = jp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -0.5, 0.0, 0.0, 0.0])

    def _append_payload_privileged(self, obs: Dict[str, jp.ndarray], payload_mass: jp.ndarray) -> Dict[str, jp.ndarray]:
        return {
            "state": obs["state"],
            "privileged_state": jp.concatenate([obs["privileged_state"], jp.array([payload_mass])], axis=-1),
            "wm_state": obs["wm_state"],
        }

    def _payload_torque_bias(self, payload_mass: jp.ndarray, step_count: jp.ndarray) -> jp.ndarray:
        active = (step_count >= self._payload_inject_step).astype(payload_mass.dtype)
        base = self._left_disturbance if self._payload_side == "left" else self._right_disturbance
        return active * self._payload_torque_scale * payload_mass * base

    def reset(self, rng: jp.ndarray) -> State:
        state = super().reset(rng)
        if self._eval_payload_mass is not None:
            payload_mass = jp.asarray(self._eval_payload_mass)
        else:
            rng, mass_rng = jax.random.split(rng)
            payload_mass = jax.random.uniform(
                mass_rng,
                shape=(),
                minval=self._payload_mass_range[0],
                maxval=self._payload_mass_range[1],
            )
        info = dict(state.info)
        info["payload_mass"] = payload_mass
        info["payload_force_z"] = -payload_mass * 9.81

        obs = self._append_payload_privileged(state.obs, payload_mass)
        metrics = dict(state.metrics)
        metrics["payload_mass"] = payload_mass
        metrics["payload_force_z"] = info["payload_force_z"]

        return state.replace(obs=obs, info=info, metrics=metrics)

    def step(self, state: State, action: jp.ndarray) -> State:
        payload_mass = state.info.get("payload_mass", jp.array(0.0))
        step_count = state.metrics.get("step_count", jp.array(0.0))
        disturbed_action = jp.clip(action + self._payload_torque_bias(payload_mass, step_count), -1.0, 1.0)

        new_state = super().step(state, disturbed_action)
        obs = self._append_payload_privileged(new_state.obs, payload_mass)

        wm_state = g1Utils.denormalize_state(new_state.obs["wm_state"], g1Utils.wm_state_limits)
        quat = wm_state[g1Utils.wm_quat_idxs] / jp.linalg.norm(wm_state[g1Utils.wm_quat_idxs])
        roll, pitch, _ = quat_to_eulerzyx(quat)
        tilt_penalty = 0.03 * (jp.abs(roll) + jp.abs(pitch))
        reward = jp.clip(new_state.reward - tilt_penalty, 0.0, 1e4)

        metrics = dict(new_state.metrics)
        metrics["payload_mass"] = payload_mass
        metrics["payload_force_z"] = -payload_mass * 9.81
        metrics["payload_tilt_penalty"] = tilt_penalty
        metrics["max_abs_roll_pitch"] = jp.maximum(jp.abs(roll), jp.abs(pitch))

        return new_state.replace(obs=obs, reward=reward, metrics=metrics)
