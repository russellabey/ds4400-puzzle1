""" Helpers for the ghost puzzle: mean TPR / FPR of estimate across lam, and a plot.

    from helper import sweep, plot, table

    results = sweep(np.linspace(11, 15, 41))   # DataFrame: lam, mean_TPR, mean_FPR
    table(results)                             # prints the rows
    plot(results, 'Mean TPR and FPR vs. lam')

TPR = (named and on the circuit) / (devices on the circuit)
FPR = (named but not on the circuit) / (devices not on the circuit)
Both are computed per circuit, then averaged over the circuits.
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ghost_circuit import GhostCircuit, OFFICIAL, estimate

# a few fits at small lam stop at the iteration limit before reaching the solver tolerance
warnings.filterwarnings('ignore')

N_ON = OFFICIAL['n_on']
N_OFF = OFFICIAL['n_dev'] - N_ON


def sweep(lams, seeds=range(100)):
    """ mean TPR and FPR of estimate(x, y, lam) at each lam

    Args:
        lams (array-like): the lam values to try
        seeds (iterable): circuit seeds, GhostCircuit.sample(seed=seed)

    Returns:
        results (pd.DataFrame): columns lam, mean_TPR, mean_FPR (one row per lam)
    """
    seeds = list(seeds)
    tpr = np.zeros((len(seeds), len(lams)))
    fpr = np.zeros((len(seeds), len(lams)))
    for si, sd in enumerate(seeds):
        circuit = GhostCircuit.sample(seed=sd)
        for li, lam in enumerate(lams):
            s = circuit.score(estimate(circuit.x, circuit.y, lam=lam))
            tpr[si, li] = s['right'] / N_ON
            fpr[si, li] = s['wrong'] / N_OFF

    return pd.DataFrame({'lam': lams,
                         'mean_TPR': tpr.mean(axis=0),
                         'mean_FPR': fpr.mean(axis=0)})


def table(results, digits=4):
    """ prints the results of sweep, one row per lam

    Args:
        results (pd.DataFrame): output of sweep
        digits (int): decimals to show
    """
    print(results.round(digits).to_string(index=False))


def plot(results, title, logx=False, ref=0.875):
    """ plots mean TPR and mean FPR against lam

    Args:
        results (pd.DataFrame): output of sweep
        title (str): plot title
        logx (bool): log-scale x-axis
        ref (float): height of the black dashed reference line (None for no line)
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(results.lam, results.mean_TPR, color='#2a6fdb', lw=2, label='mean TPR')
    ax.plot(results.lam, results.mean_FPR, color='#e8833a', lw=2, label='mean FPR')
    if logx:
        ax.set_xscale('log')
    ax.set_xlabel('lam')
    ax.set_ylabel('rate')
    ax.set_ylim(-.02, 1.02)
    if ref is not None:
        ax.axhline(ref, color='black', ls='--', lw=1.2, label=str(ref))
    ax.set_title(title)
    ax.grid(alpha=.25)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()
    return fig, ax
