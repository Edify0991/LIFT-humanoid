#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import dill
import jax
import jax.numpy as jp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lift_utils import running_statistics
from policy_pretrain import sac_networks
from brax.math import quat_to_eulerzyx
from brax.robots.g1.utils import g1Utils
from brax.envs.g1_lowdim_payload_walking import G1LowDimPayloadWalking


def _unpack_policy_state(sac_ts):
    if hasattr(sac_ts, "policy_params") and hasattr(sac_ts, "normalizer_params"):
        return sac_ts.normalizer_params, sac_ts.policy_params
    raise ValueError("Unsupported policy checkpoint format.")


def build_policy(env, normalizer_params, policy_params):
    sac_network = sac_networks.make_sac_networks(
        observation_size=env.observation_size,
        action_size=env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=(512, 256, 128),
        q_hidden_layer_sizes=(1024, 512, 256),
        activation='swish',
    )

    def act(obs):
        logits = sac_network.policy_network.apply(normalizer_params, policy_params, obs)
        return sac_network.parametric_action_distribution.mode(logits)

    return jax.jit(act)


def run_episode(env, policy_fn, seed):
    key = jax.random.PRNGKey(seed)
    state = env.reset(key)
    traj_q, traj_qd, traj_a = [], [], []
    max_roll_pitch = 0.0
    energy = 0.0
    done_step = 0
    for t in range(1000):
        a = policy_fn(state.obs)
        state = env.step(state, a)
        wm = g1Utils.denormalize_state(state.obs['wm_state'], g1Utils.wm_state_limits)
        quat = wm[g1Utils.wm_quat_idxs] / jp.linalg.norm(wm[g1Utils.wm_quat_idxs])
        roll, pitch, _ = quat_to_eulerzyx(quat)
        max_roll_pitch = max(max_roll_pitch, float(jp.maximum(jp.abs(roll), jp.abs(pitch))))
        energy += float(jp.sum(jp.abs(state.torque * wm[g1Utils.wm_qd_idxs])))
        traj_q.append(np.array(wm[g1Utils.wm_q_idxs]))
        traj_qd.append(np.array(wm[g1Utils.wm_qd_idxs]))
        traj_a.append(np.array(a))
        done_step = t + 1
        if float(state.done) > 0.5:
            break
    success = float(done_step >= 800)
    return success, done_step, max_roll_pitch, energy, np.array(traj_q), np.array(traj_qd), np.array(traj_a)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--policy_ckpt', required=True)
    p.add_argument('--masses', default='0,2,4,6,8,10')
    p.add_argument('--num_episodes', type=int, default=5)
    p.add_argument('--out_dir', default='logs/payload_eval')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.policy_ckpt, 'rb') as f:
        ts = dill.load(f)
    normalizer_params, policy_params = _unpack_policy_state(ts)

    rows = []
    masses = [float(x) for x in args.masses.split(',') if x]
    for m in masses:
        env = G1LowDimPayloadWalking(eval_payload_mass=m, backend='generalized', forces_in_q_coords=True)
        policy_fn = build_policy(env, normalizer_params, policy_params)
        succ, falls, rps, ens = [], [], [], []
        for ep in range(args.num_episodes):
            s, fall_t, rp, en, q, qd, a = run_episode(env, policy_fn, ep)
            succ.append(s); falls.append(fall_t); rps.append(rp); ens.append(en)
            np.savez(out_dir / f'traj_m{m:.2f}_ep{ep}.npz', q=q, qd=qd, action=a)
        rows.append(dict(payload_mass=m, success_rate=np.mean(succ), avg_fall_time=np.mean(falls), max_roll_pitch=np.mean(rps), energy=np.mean(ens)))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'payload_sweep.csv', index=False)

    plt.figure(figsize=(6,4))
    plt.plot(df['payload_mass'], df['success_rate'], marker='o')
    plt.xlabel('Payload mass (kg)')
    plt.ylabel('Success rate')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / 'success_vs_mass.png', dpi=150)


if __name__ == '__main__':
    main()
