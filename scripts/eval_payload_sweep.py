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
    if isinstance(sac_ts, dict) and "policy_params" in sac_ts and "normalizer_params" in sac_ts:
        return sac_ts["normalizer_params"], sac_ts["policy_params"]
    raise ValueError("Unsupported policy checkpoint format.")


def build_policy(env, normalizer_params, policy_params):
    sac_network = sac_networks.make_sac_networks(
        observation_size=env.observation_size,
        action_size=env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=(512, 256, 128),
        q_hidden_layer_sizes=(1024, 512, 256),
        activation='swish',
        policy_obs_key='state',
        value_obs_key='privileged_state',
    )

    def act(obs):
        logits = sac_network.policy_network.apply(normalizer_params, policy_params, obs)
        return sac_network.parametric_action_distribution.mode(logits)

    return jax.jit(act)


def run_episode(env, policy_fn, seed, horizon=1000):
    key = jax.random.PRNGKey(seed)
    state = env.reset(key)
    traj_q, traj_qd, traj_a = [], [], []
    max_roll_pitch = 0.0
    energy = 0.0
    done_step = 0
    for t in range(horizon):
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
    success = float(done_step >= int(0.8 * horizon))
    fall = float(done_step < int(0.8 * horizon))
    settle_time = float(done_step)
    return {
        'success': success,
        'fall': fall,
        'done_step': done_step,
        'max_roll_pitch': max_roll_pitch,
        'energy': energy,
        'settle_time': settle_time,
        'q': np.array(traj_q),
        'qd': np.array(traj_qd),
        'action': np.array(traj_a),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--env_name', default='G1LowDimPayloadWalking')
    p.add_argument('--policy_ckpt', required=True)
    p.add_argument('--baseline_name', default='Ours')
    p.add_argument('--masses', default='0,2,4,6,8,10')
    p.add_argument('--num_episodes', type=int, default=5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--horizon', type=int, default=1000)
    p.add_argument('--out_dir', default='logs/payload_eval')
    args = p.parse_args()

    if args.env_name != 'G1LowDimPayloadWalking':
        raise ValueError('MVP supports env_name=G1LowDimPayloadWalking')

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
        episode_stats = []
        for ep in range(args.num_episodes):
            stat = run_episode(env, policy_fn, args.seed + ep, horizon=args.horizon)
            episode_stats.append(stat)
            np.savez(
                out_dir / f'traj_{args.baseline_name}_m{m:.2f}_ep{ep}.npz',
                q=stat['q'],
                qd=stat['qd'],
                action=stat['action'],
            )

        rows.append(dict(
            baseline=args.baseline_name,
            payload_mass=m,
            success_rate=float(np.mean([x['success'] for x in episode_stats])),
            fall_rate=float(np.mean([x['fall'] for x in episode_stats])),
            avg_fall_time=float(np.mean([x['done_step'] for x in episode_stats])),
            max_roll_pitch=float(np.mean([x['max_roll_pitch'] for x in episode_stats])),
            energy=float(np.mean([x['energy'] for x in episode_stats])),
            convergence_time=float(np.mean([x['settle_time'] for x in episode_stats])),
        ))

    df = pd.DataFrame(rows)
    csv_path = out_dir / f'payload_sweep_{args.baseline_name}.csv'
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(6, 4))
    plt.plot(df['payload_mass'], df['success_rate'], marker='o', label=args.baseline_name)
    plt.xlabel('Payload mass (kg)')
    plt.ylabel('Success rate')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f'success_vs_mass_{args.baseline_name}.png', dpi=150)
    print(f'Saved: {csv_path}')


if __name__ == '__main__':
    main()
