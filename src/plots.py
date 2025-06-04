"""Plot helpers – all figures in one place."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

_palette = dict(fast="tab:orange", normal="tab:blue", slow="tab:red")

def plot_jitter(mean, sem, label, ax=None):
    if ax is None:
        ax = plt.gca()
    idx = np.arange(len(mean))
    ax.fill_between(idx, mean - sem, mean + sem,
                    color=_palette[label], alpha=0.2)
    ax.plot(idx, mean, color=_palette[label], label=label.capitalize())

def boxplot(ax, data, labels):
    ax.boxplot(data, labels=labels)
    ax.set_ylabel(r"|$\Delta t$|  (s)")

def plot_phase(mean, sem, label, ax=None):
    if ax is None:
        ax = plt.gca()
    idx = np.arange(len(mean))
    ax.fill_between(idx, mean - sem, mean + sem,
                    color=_palette[label], alpha=0.2)
    ax.plot(idx, mean, color=_palette[label], label=label.capitalize())

def plot_ccg(ax, centers, early, late, label):
    ax.plot(centers, early, ls="--", color=_palette[label], label="Early")
    ax.plot(centers, late,  ls="-",  color=_palette[label], label="Late")
    ax.set_title(f"{label.capitalize()} memresistor")
    ax.set_xlabel("Lag B – A (ms)")
    ax.legend()
