#!/usr/bin/env python3
"""
FINAL CORRECTED plot2.py that uses properly computed z-scores.

This version uses z-scores computed from:
  z = calibrated_effect / sqrt(position_bias)

Where position_bias is the VARIANCE of forced SSIs.
"""

import argparse
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Define consistent color schemes
MODEL_FAMILY_COLORS = {
    "Anthropic": "#E74C3C",  # Red
    "OpenAI": "#3498DB",  # Blue
    "Google": "#2ECC71",  # Green
    "OSS": "#F39C12",  # Orange
    "Other": "#9B59B6"  # Purple
}

DOMAIN_COLORS = {
    "harmfulness": "#1ABC9C",  # Teal
    "correctness": "#E67E22",  # Dark Orange
    "unknown": "#95A5A6"  # Gray
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", default="samples_with_computed_zscores.csv",
                   help="Path to sample-level CSV with z-scores")
    p.add_argument("--tasks", default="tasks_with_computed_zscores.csv",
                   help="Path to task-level aggregated CSV")
    p.add_argument("--out", default="plots", help="Output directory for charts")
    p.add_argument("--dpi", type=int, default=180, help="Figure DPI")
    p.add_argument("--style", default="whitegrid", help="Seaborn style")
    return p.parse_args()


def bar_calibrated_by_task_with_significance(tasks_df, outdir, dpi):
    """Bar chart showing calibrated effects with proper significance from z-scores."""

    # Sort by mean calibrated effect
    tasks_df = tasks_df.sort_values('score_calibrated_effect_scorer_mean')

    # Determine colors based on significance rate
    colors = []
    for _, row in tasks_df.iterrows():
        sig_rate = row.get('pct_significant', 0)
        if sig_rate >= 75:
            colors.append('#2ECC71')  # Green - highly significant
        elif sig_rate >= 50:
            colors.append('#3498DB')  # Blue - moderately significant
        elif sig_rate >= 25:
            colors.append('#F39C12')  # Orange - some significance
        else:
            colors.append('#95A5A6')  # Gray - mostly not significant

    plt.figure(figsize=(12, max(6, 0.3 * len(tasks_df))))
    ax = plt.gca()

    y_pos = np.arange(len(tasks_df))

    # Use within-sample std as error bars
    errors = tasks_df['within_sample_std_mean'].fillna(0).values

    bars = ax.barh(y_pos, tasks_df['score_calibrated_effect_scorer_mean'],
                   xerr=errors, color=colors, alpha=0.8,
                   error_kw={'elinewidth': 1, 'capsize': 3, 'alpha': 0.6})

    ax.axvline(0, color="k", lw=1, alpha=0.6)
    ax.set_yticks(y_pos)

    # Create labels showing task and significance
    labels = []
    for _, row in tasks_df.iterrows():
        domain = row.get('domain', 'unk')
        dataset = row.get('dataset', 'unk')
        sig_rate = row.get('pct_significant', 0)
        labels.append(f"{domain[:4]}/{dataset[:15]} ({sig_rate:.0f}%)")

    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Calibrated effect (SSI_actual − mean(SSI_forced))", fontsize=11)
    ax.set_title("Calibrated Effects with Within-Sample Error Bars\n"
                 "Color indicates % samples significant", fontsize=13, fontweight='bold')

    # Add value annotations
    for i, (_, row) in enumerate(tasks_df.iterrows()):
        val = row['score_calibrated_effect_scorer_mean']
        mean_z = row.get('z_score_mean', 0)

        # Determine significance marker based on mean z-score
        if abs(mean_z) > 3.29:
            sig = "***"
        elif abs(mean_z) > 2.58:
            sig = "**"
        elif abs(mean_z) > 1.96:
            sig = "*"
        else:
            sig = ""

        ax.text(val + (0.02 if val >= 0 else -0.02), i,
                f"{val:.3f}{sig}", va="center",
                ha="left" if val >= 0 else "right",
                fontsize=8, fontweight='bold' if sig else 'normal')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ECC71", alpha=0.8, label="≥75% samples significant"),
        Patch(facecolor="#3498DB", alpha=0.8, label="50-75% significant"),
        Patch(facecolor="#F39C12", alpha=0.8, label="25-50% significant"),
        Patch(facecolor="#95A5A6", alpha=0.8, label="<25% significant")
    ]
    ax.legend(handles=legend_elements, loc="best", fontsize=8)

    # Add statistics note
    ax.text(0.02, 0.02,
            "Error bars: ±1 within-sample SD\n* p<0.05, ** p<0.01, *** p<0.001 (mean z)",
            transform=ax.transAxes, fontsize=7, alpha=0.6, va='bottom')

    plt.tight_layout()
    plt.savefig(Path(outdir) / "calibrated_effects_by_task.png", dpi=dpi, bbox_inches='tight')
    plt.close()


def scatter_zscore_analysis(samples_df, outdir, dpi):
    """Create scatter plots analyzing z-scores."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Z-score vs Effect Size
    ax = axes[0, 0]

    # Color by significance
    colors = ['green' if abs(z) > 1.96 else 'gray'
              for z in samples_df['z_score']]

    ax.scatter(samples_df['score_calibrated_effect_scorer'],
               samples_df['z_score'],
               c=colors, alpha=0.3, s=10)

    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(-1.96, color='red', linestyle='--', alpha=0.5, label='p=0.05')
    ax.axhline(1.96, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Calibrated Effect', fontsize=11)
    ax.set_ylabel('Z-score', fontsize=11)
    ax.set_title('Z-score vs Effect Size', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)

    # Add correlation
    corr = samples_df[['score_calibrated_effect_scorer', 'z_score']].corr().iloc[0, 1]
    ax.text(0.02, 0.98, f'Correlation: {corr:.3f}',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 2. Within-sample STD vs Effect
    ax = axes[0, 1]
    ax.scatter(samples_df['within_sample_std'],
               samples_df['score_calibrated_effect_scorer'].abs(),
               alpha=0.3, s=10, color='blue')
    ax.set_xlabel('Within-sample STD', fontsize=11)
    ax.set_ylabel('|Calibrated Effect|', fontsize=11)
    ax.set_title('Effect Size vs Uncertainty', fontsize=12, fontweight='bold')

    # 3. Distribution of significance by domain
    ax = axes[1, 0]
    if 'domain' in samples_df.columns:
        domain_data = []
        domains = []
        for domain in sorted(samples_df['domain'].dropna().unique()):
            domain_samples = samples_df[samples_df['domain'] == domain]
            if len(domain_samples) > 0:
                sig_rate = (domain_samples['significant'] == True).mean() * 100
                domain_data.append(sig_rate)
                domains.append(domain)

        if domain_data:
            colors = [DOMAIN_COLORS.get(d, '#95A5A6') for d in domains]
            bars = ax.bar(range(len(domain_data)), domain_data, color=colors, alpha=0.8)
            ax.set_xticks(range(len(domain_data)))
            ax.set_xticklabels(domains, fontsize=10)
            ax.set_ylabel('% Samples Significant', fontsize=11)
            ax.set_title('Significance Rate by Domain', fontsize=12, fontweight='bold')
            ax.axhline(5, color='black', linestyle='--', alpha=0.3,
                       label='5% (expected under null)')
            ax.legend(fontsize=8)

            # Add value labels
            for i, (rate, domain) in enumerate(zip(domain_data, domains)):
                n = len(samples_df[samples_df['domain'] == domain])
                ax.text(i, rate + 1, f"{rate:.1f}%\n(n={n})",
                        ha='center', va='bottom', fontsize=8)

    # 4. Top datasets by significance
    ax = axes[1, 1]
    if 'dataset' in samples_df.columns:
        dataset_sig = samples_df.groupby('dataset').agg({
            'significant': lambda x: (x == True).mean() * 100,
            'score_calibrated_effect_scorer': 'mean'
        }).sort_values('significant', ascending=False).head(10)

        y_pos = range(len(dataset_sig))
        colors = ['green' if sig > 50 else 'orange' if sig > 25 else 'gray'
                  for sig in dataset_sig['significant']]

        bars = ax.barh(y_pos, dataset_sig['significant'], color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([d[:20] for d in dataset_sig.index], fontsize=8)
        ax.set_xlabel('% Samples Significant', fontsize=11)
        ax.set_title('Top 10 Datasets by Significance Rate', fontsize=12, fontweight='bold')
        ax.axvline(5, color='black', linestyle='--', alpha=0.3)

        # Add value labels
        for i, (idx, row) in enumerate(dataset_sig.iterrows()):
            ax.text(row['significant'] + 1, i, f"{row['significant']:.0f}%",
                    va='center', fontsize=8)

    plt.suptitle('Z-score and Significance Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(Path(outdir) / "zscore_analysis.png", dpi=dpi, bbox_inches='tight')
    plt.close()


def histogram_distributions(samples_df, outdir, dpi):
    """Create histograms of key distributions."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Z-score distribution
    ax = axes[0, 0]
    ax.hist(samples_df['z_score'].dropna(), bins=50, edgecolor='black',
            alpha=0.7, color='#3498DB')
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.axvline(-1.96, color='red', linestyle='--', alpha=0.5)
    ax.axvline(1.96, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Z-score', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Distribution of Z-scores', fontsize=12, fontweight='bold')

    # Add statistics
    z_scores = samples_df['z_score'].dropna()
    sig_rate = (z_scores.abs() > 1.96).mean() * 100
    ax.text(0.02, 0.98,
            f'N = {len(z_scores)}\n'
            f'Mean = {z_scores.mean():.3f}\n'
            f'SD = {z_scores.std():.3f}\n'
            f'Significant = {sig_rate:.1f}%',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 2. Calibrated effect distribution
    ax = axes[0, 1]
    effects = samples_df['score_calibrated_effect_scorer'].dropna()
    ax.hist(effects, bins=40, edgecolor='black', alpha=0.7, color='#E67E22')
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Calibrated Effect', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Distribution of Calibrated Effects', fontsize=12, fontweight='bold')

    ax.text(0.02, 0.98,
            f'N = {len(effects)}\n'
            f'Mean = {effects.mean():.4f}\n'
            f'SD = {effects.std():.4f}',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 3. Within-sample STD distribution
    ax = axes[1, 0]
    stds = samples_df['within_sample_std'].dropna()
    ax.hist(stds, bins=30, edgecolor='black', alpha=0.7, color='#9B59B6')
    ax.set_xlabel('Within-sample STD', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Distribution of Within-Sample STDs', fontsize=12, fontweight='bold')

    ax.text(0.98, 0.98,
            f'Mean = {stds.mean():.3f}\n'
            f'Median = {stds.median():.3f}\n'
            f'SD = {stds.std():.3f}',
            transform=ax.transAxes, va='top', ha='right', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 4. SSI distribution
    ax = axes[1, 1]
    ssi = samples_df['score_comprehensive_ssi_scorer'].dropna()
    ax.hist(ssi, bins=40, edgecolor='black', alpha=0.7, color='#2ECC71')
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('SSI (prefilled actual)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Distribution of SSI Scores', fontsize=12, fontweight='bold')

    ax.text(0.02, 0.98,
            f'N = {len(ssi)}\n'
            f'Mean = {ssi.mean():.3f}\n'
            f'SD = {ssi.std():.3f}',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Distribution Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(Path(outdir) / "distributions.png", dpi=dpi, bbox_inches='tight')
    plt.close()


def print_summary(samples_df, tasks_df):
    """Print comprehensive summary statistics."""

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS - PROPER WITHIN-SAMPLE SIGNIFICANCE")
    print("=" * 80)

    # Sample-level statistics
    z_scores = samples_df['z_score'].dropna()
    if len(z_scores) > 0:
        print("\n=== Sample-Level Statistics ===")
        print(f"Total samples with z-scores: {len(z_scores)}")
        print(f"Mean z-score: {z_scores.mean():.3f}")
        print(f"SD of z-scores: {z_scores.std():.3f}")

        sig_pos = (z_scores > 1.96).sum()
        sig_neg = (z_scores < -1.96).sum()
        print(f"\nSignificant positive: {sig_pos} ({100 * sig_pos / len(z_scores):.1f}%)")
        print(f"Significant negative: {sig_neg} ({100 * sig_neg / len(z_scores):.1f}%)")
        print(f"Not significant: {len(z_scores) - sig_pos - sig_neg} "
              f"({100 * (len(z_scores) - sig_pos - sig_neg) / len(z_scores):.1f}%)")

    # Task-level statistics
    if not tasks_df.empty and 'pct_significant' in tasks_df.columns:
        print("\n=== Task-Level Statistics ===")
        print(f"Total tasks: {len(tasks_df)}")
        print(f"Mean significance rate: {tasks_df['pct_significant'].mean():.1f}%")
        print(f"Median significance rate: {tasks_df['pct_significant'].median():.1f}%")

        print("\nTop 5 tasks by significance rate:")
        top_tasks = tasks_df.nlargest(5, 'pct_significant')
        for _, row in top_tasks.iterrows():
            print(f"  {row['domain']}/{row['dataset'][:20]:20s}: "
                  f"{row['pct_significant']:5.1f}% sig, "
                  f"effect={row['score_calibrated_effect_scorer_mean']:7.4f}")

    # Domain comparison
    if 'domain' in samples_df.columns:
        print("\n=== Domain Comparison ===")
        for domain in sorted(samples_df['domain'].dropna().unique()):
            domain_data = samples_df[samples_df['domain'] == domain]
            z_domain = domain_data['z_score'].dropna()
            if len(z_domain) > 0:
                sig_rate = (z_domain.abs() > 1.96).mean() * 100
                print(f"{domain:15s}: {sig_rate:5.1f}% significant, "
                      f"mean z={z_domain.mean():6.3f}, "
                      f"n={len(z_domain)}")

    print("\n" + "=" * 80)
    print("KEY INSIGHT: We use z = calibrated_effect / sqrt(position_bias)")
    print("This works because position_bias is the VARIANCE of forced SSIs")
    print("=" * 80)


def main():
    args = parse_args()

    # Create output directory
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")

    # Load sample-level data
    if Path(args.samples).exists():
        samples_df = pd.read_csv(args.samples)
        print(f"  Loaded {len(samples_df)} samples from {args.samples}")
    else:
        print(f"ERROR: Sample file not found: {args.samples}")
        print("Please run compute_zscore_from_position_bias.py first!")
        return

    # Load task-level data
    if Path(args.tasks).exists():
        tasks_df = pd.read_csv(args.tasks)
        print(f"  Loaded {len(tasks_df)} tasks from {args.tasks}")
    else:
        print(f"WARNING: Task file not found: {args.tasks}")
        tasks_df = pd.DataFrame()

    # Set style
    sns.set_style(args.style)

    # Generate plots
    print("\nGenerating plots...")

    if not tasks_df.empty:
        bar_calibrated_by_task_with_significance(tasks_df, outdir, args.dpi)
        print("  ✓ Task-level bar chart")

    scatter_zscore_analysis(samples_df, outdir, args.dpi)
    print("  ✓ Z-score scatter plots")

    histogram_distributions(samples_df, outdir, args.dpi)
    print("  ✓ Distribution histograms")

    # Print summary
    print_summary(samples_df, tasks_df)

    print(f"\nAll plots saved to: {outdir}")
    print("\nDone! The plots now show PROPER within-sample significance.")


if __name__ == "__main__":
    main()