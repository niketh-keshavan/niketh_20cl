"""
Command-line entry point:

python -m src.run_analysis --datadir data --outdir figs
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .io import load_groups
from .metrics import (early_late_stats, jitter_curves, phase_curves,
                      cross_correlogram, paired_t)
from .plots import plot_jitter, boxplot, plot_phase, plot_ccg

# ---------- mapping of filenames ----------
FILES = {
    "fast":   [f"spike_data_fast{i}.csv"  for i in range(1, 6)],
    "normal": [f"spike_data{i}.csv"       for i in range(1, 6)],
    "slow":   [f"spike_data_slow{i}.csv"  for i in range(1, 3)],
}

# ---------- main ----------
def main(datadir: Path, outdir: Path):
    outdir.mkdir(exist_ok=True)
    groups = load_groups(FILES, datadir)

    # --- Jitter curves & boxplot ---
    fig_jit, ax_jit = plt.subplots(figsize=(8, 5))
    box_data, box_labels = [], []
    summary_rows = []

    for label, dfs in groups.items():
        dts = [df["time_B_s"].values - df["time_A_s"].values for df in dfs]
        mean, sem = jitter_curves(dts)
        plot_jitter(mean, sem, label, ax=ax_jit)

        # early/late stats
        early = np.array([early_late_stats(np.abs(dt))[0] for dt in dts])
        late  = np.array([early_late_stats(np.abs(dt))[1] for dt in dts])
        box_data.extend([early, late])
        box_labels.extend([f"{label}\nEarly", f"{label}\nLate"])
        t, p = paired_t(early, late)
        summary_rows.append(dict(group=label, early=early.mean(),
                                 late=late.mean(), t=t, p=p))

    ax_jit.set_xlabel("Spike index (smoothed, window = 100)")
    ax_jit.set_ylabel(r"Mean |$\Delta t$|  (s)")
    ax_jit.set_title("Figure 2A – Convergence of spike-time jitter")
    ax_jit.legend()
    fig_jit.tight_layout()
    fig_jit.savefig(outdir / "fig2A_jitter.png", dpi=300)

    # boxplot
    fig_box, ax_box = plt.subplots(figsize=(8, 5))
    boxplot(ax_box, box_data, box_labels)
    ax_box.set_title("Figure 2B – Early vs late spike-time jitter")
    fig_box.tight_layout()
    fig_box.savefig(outdir / "fig2B_box.png", dpi=300)

    # --- Phase curves ---
    fig_phi, ax_phi = plt.subplots(figsize=(8, 5))
    for label, dfs in groups.items():
        dts = [df["time_B_s"].values - df["time_A_s"].values for df in dfs]
        mean, sem = phase_curves(dts)
        plot_phase(mean, sem, label, ax=ax_phi)
    ax_phi.set_xlabel("Spike index (smoothed, window = 100)")
    ax_phi.set_ylabel("Resultant R")
    ax_phi.set_title("Figure 3 – Evolution of phase locking")
    ax_phi.legend()
    fig_phi.tight_layout()
    fig_phi.savefig(outdir / "fig3_phase.png", dpi=300)

    # --- Cross-correlograms ---
    bins = np.arange(-0.1, 0.105, 0.005)
    centers = (bins[:-1] + bins[1:]) / 2 * 1000
    fig_ccg, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, label in zip(axes, ["slow", "normal", "fast"]):
        dfs = groups[label]
        tA_all, tB_all = [], []
        for df in dfs:
            tA_all.append(df["time_A_s"].values)
            tB_all.append(df["time_B_s"].values)
        tA = np.concatenate(tA_all)
        tB = np.concatenate(tB_all)
        n = len(tA)
        hE = cross_correlogram(tA[: n // 5], tB[: n // 5], bins)
        hL = cross_correlogram(tA[-n // 5 :], tB[-n // 5 :], bins)
        plot_ccg(ax, centers, hE, hL, label)
        if ax is axes[0]:
            ax.set_ylabel("Density")
    fig_ccg.suptitle("Figure 4 – Cross-correlogram sharpening")
    fig_ccg.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_ccg.savefig(outdir / "fig4_ccg.png", dpi=300)

    # --- stats table ---
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(outdir / "stats_summary.csv", index=False)
    print(df_summary)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datadir", type=str, default="data",
                   help="Directory containing the CSV spike files")
    p.add_argument("--outdir", type=str, default="figs",
                   help="Where to save figures & summary CSV")
    args = p.parse_args()
    main(Path(args.datadir), Path(args.outdir))
