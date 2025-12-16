#!/usr/bin/env python3
"""
Create conceptual diagrams for False Positive (FP) and False Negative (FN) context signal patterns.
Saves figures for presentation in house style.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_fp(ax):
    x = np.linspace(-4, 4, 200)
    # Broad, smooth hill (FP)
    signal = 0.8 * np.exp(-x**2/3) + 0.15
    ax.plot(x, signal, color='red', lw=4)
    ax.fill_between(x, 0, signal, color='red', alpha=0.3)
    ax.axvline(0, color='red', ls='--', lw=2)
    # Title at top
    ax.set_title('FALSE POSITIVE\n(Broad Signal)', color='red', fontsize=14, fontweight='bold', pad=12)
    ax.text(-3.5, 0.7, 'Peak Height Ratio = 2.2', bbox=dict(facecolor='red', alpha=0.2), fontsize=12)
    ax.text(0, 0.82, 'Predicted Site', ha='center', va='bottom', fontsize=13, fontweight='bold', color='black')
    ax.set_xlabel('Position relative to splice site')
    ax.set_ylabel('Probability Score')
    ax.set_ylim(0, 1.2)
    ax.set_xlim(-4, 4)
    ax.grid(alpha=0.3)
    ax.set_xticks(np.arange(-4, 5, 1))

def plot_fn(ax):
    x = np.linspace(-4, 4, 200)
    # Blunted/flat at 0, with neighbors nearly as high (FN)
    signal = 0.4 * np.exp(-((x-1.2)**2)/0.5) + 0.4 * np.exp(-((x+1.2)**2)/0.5) + 0.2
    ax.plot(x, signal, color='orange', lw=4)
    ax.fill_between(x, 0, signal, color='orange', alpha=0.3)
    ax.axvline(0, color='orange', ls='--', lw=2)
    # Title at top
    ax.set_title('FALSE NEGATIVE\n(Blunted/Flat at True Site)', color='orange', fontsize=14, fontweight='bold', pad=12)
    ax.text(-3.5, 0.7, 'Peak Height Ratio ≈ 1.1', bbox=dict(facecolor='orange', alpha=0.2), fontsize=12)
    ax.text(0, 0.45, 'True Site (Missed)', ha='center', va='bottom', fontsize=13, fontweight='bold', color='black')
    ax.set_xlabel('Position relative to splice site')
    ax.set_ylabel('Probability Score')
    ax.set_ylim(0, 1.2)
    ax.set_xlim(-4, 4)
    ax.grid(alpha=0.3)
    ax.set_xticks(np.arange(-4, 5, 1))

def main():
    outdir = Path('results/probability_feature_analysis/quick/diagrams')
    outdir.mkdir(parents=True, exist_ok=True)

    # FP only
    fig, ax = plt.subplots(figsize=(7,5))
    plot_fp(ax)
    plt.tight_layout()
    plt.savefig(outdir / 'fp_concept_signal.png', dpi=300)
    plt.close()

    # FN only
    fig, ax = plt.subplots(figsize=(7,5))
    plot_fn(ax)
    plt.tight_layout()
    plt.savefig(outdir / 'fn_concept_signal.png', dpi=300)
    plt.close()

    # Side-by-side for presentation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13,5), sharey=True)
    plot_fp(ax1)
    plot_fn(ax2)
    plt.tight_layout()
    plt.savefig(outdir / 'fp_fn_concept_signal_side_by_side.png', dpi=300)
    plt.close()

    print(f"Saved conceptual FP/FN signal diagrams to {outdir}")

if __name__ == '__main__':
    main() 