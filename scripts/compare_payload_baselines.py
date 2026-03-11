#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csvs', required=True, help='comma-separated payload_sweep_*.csv files')
    p.add_argument('--out_dir', default='logs/payload_eval')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = [Path(x) for x in args.csvs.split(',') if x]
    dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(out_dir / 'payload_baselines_merged.csv', index=False)

    plt.figure(figsize=(7, 4))
    for baseline, g in df.groupby('baseline'):
        g = g.sort_values('payload_mass')
        plt.plot(g['payload_mass'], g['success_rate'], marker='o', label=baseline)
    plt.xlabel('Payload mass (kg)')
    plt.ylabel('Success rate')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'payload_baselines_success.png', dpi=150)


if __name__ == '__main__':
    main()
