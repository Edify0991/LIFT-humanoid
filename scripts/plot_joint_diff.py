#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def l2_seq(a, b):
    t = min(len(a), len(b))
    return float(np.mean(np.linalg.norm(a[:t] - b[:t], axis=-1)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--traj_dir', required=True)
    p.add_argument('--baseline_mass', type=float, default=0.0)
    p.add_argument('--out_csv', default='joint_diff.csv')
    args = p.parse_args()

    traj_dir = Path(args.traj_dir)
    files = sorted(traj_dir.glob('traj_m*.npz'))
    baseline = [f for f in files if f'traj_m{args.baseline_mass:.2f}_' in f.name]
    if not baseline:
        raise FileNotFoundError('No baseline trajectory found.')
    q_base = np.load(baseline[0])['q']

    rows = []
    for f in files:
        q = np.load(f)['q']
        mass = float(f.name.split('_')[1][1:])
        rows.append({'file': f.name, 'payload_mass': mass, 'q_l2_vs_base': l2_seq(q, q_base)})

    df = pd.DataFrame(rows).sort_values('payload_mass')
    df.to_csv(traj_dir / args.out_csv, index=False)

    plt.figure(figsize=(6,4))
    plt.scatter(df['payload_mass'], df['q_l2_vs_base'])
    plt.xlabel('Payload mass (kg)')
    plt.ylabel('Joint trajectory L2 vs baseline')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(traj_dir / 'joint_diff.png', dpi=150)


if __name__ == '__main__':
    main()
