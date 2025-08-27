#!/usr/bin/env python3
"""
Simplified version that directly uses eval_id to group data and create plots.
This should definitely work with your data structure.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

# Color schemes
SIGNIFICANCE_COLORS = {
    "high": "#2ECC71",      # Green - strong significance
    "moderate": "#3498DB",   # Blue - moderate significance
    "low": "#F39C12",       # Orange - weak significance
    "none": "#95A5A6"       # Gray - not significant
}


def create_calibrated_plots_simple(csv_path, output_dir="plots_with_errorbars", dpi=180):
    """
    Create calibrated effect plots using eval_id as the grouping variable.
    This is simpler and more robust.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} samples")

    # Check what columns we have
    print("\nChecking data structure:")
    print(f"  Columns: {df.columns.tolist()[:10]}...")  # Show first 10 columns

    # Group by domain and dataset
    if 'domain' not in df.columns or 'dataset' not in df.columns:
        print("ERROR: Missing domain or dataset columns!")
        return

    unique_combos = df.groupby(['domain', 'dataset']).size().reset_index(name='count')
    print(f"\nFound {len(unique_combos)} domain/dataset combinations:")
    for _, row in unique_combos.iterrows():
        print(f"  {row['domain']}/{row['dataset']}: {row['count']} samples")

    # Create plots for each domain/dataset combination
    for _, combo in unique_combos.iterrows():
        domain = combo['domain']
        dataset = combo['dataset']

        # Skip if invalid
        if pd.isna(domain) or pd.isna(dataset):
            continue

        print(f"\nProcessing {domain}/{dataset}...")

        # Filter data
        subset = df[(df['domain'] == domain) & (df['dataset'] == dataset)].copy()

        # Group by eval_id (each eval_id represents a different model run)
        # This is the simplest approach that should always work
        grouped = subset.groupby('eval_id').agg({
            'score_calibrated_effect_scorer': 'mean',
            'within_sample_std': 'mean',
            'z_score': ['mean', lambda x: (x.abs() > 1.96).mean() * 100]
        }).reset_index()

        # Flatten column names
        grouped.columns = ['eval_id', 'effect_mean', 'within_std_mean',
                           'z_mean', 'pct_significant']

        # Count samples per eval_id
        counts = subset.groupby('eval_id').size().reset_index(name='n_samples')
        grouped = grouped.merge(counts, on='eval_id')

        # Sort by effect size
        grouped = grouped.sort_values('effect_mean')

        print(f"  Created {len(grouped)} aggregated entries")

        if len(grouped) == 0:
            print(f"  SKIPPING: No data after aggregation")
            continue

        # Create figure
        fig, ax = plt.subplots(figsize=(12, max(6, 0.4 * len(grouped))))

        y_pos = np.arange(len(grouped))

        # Determine colors based on significance
        colors = []
        for _, row in grouped.iterrows():
            pct_sig = row['pct_significant']
            if pct_sig >= 50:
                colors.append(SIGNIFICANCE_COLORS['high'])
            elif pct_sig >= 25:
                colors.append(SIGNIFICANCE_COLORS['moderate'])
            elif pct_sig >= 10:
                colors.append(SIGNIFICANCE_COLORS['low'])
            else:
                colors.append(SIGNIFICANCE_COLORS['none'])

        # Create horizontal bar chart with error bars
        # THIS IS THE KEY PART - we're creating bars with error bars
        ax.barh(y_pos, grouped['effect_mean'].values,
                xerr=grouped['within_std_mean'].values,
                color=colors, alpha=0.8,
                error_kw={'elinewidth': 1.5, 'capsize': 4, 'alpha': 0.7,
                          'capthick': 1.5, 'ecolor': 'black'})

        # Add vertical line at zero
        ax.axvline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.8)

        # Set y-axis labels
        ax.set_yticks(y_pos)

        # Create shorter labels from eval_id
        labels = []
        for _, row in grouped.iterrows():
            # Extract meaningful part from eval_id
            eval_id = str(row['eval_id'])
            # Try to extract model name from the path
            if '/' in eval_id:
                parts = eval_id.split('/')
                model_part = parts[-1] if parts[-1] else parts[-2]
            else:
                model_part = eval_id

            # Further clean up
            if '_' in model_part:
                model_part = model_part.split('_')[0]

            # Truncate if too long
            model_display = model_part[:30]
            label = f"{model_display} (n={row['n_samples']:.0f})"
            labels.append(label)

        ax.set_yticklabels(labels, fontsize=8)

        # Add value annotations
        for i, (_, row) in enumerate(grouped.iterrows()):
            val = row['effect_mean']
            z = row['z_mean']
            pct_sig = row['pct_significant']

            # Determine significance marker based on z-score
            if abs(z) > 3.29:
                sig_marker = "***"
            elif abs(z) > 2.58:
                sig_marker = "**"
            elif abs(z) > 1.96:
                sig_marker = "*"
            else:
                sig_marker = ""

            # Position text to the right of positive bars, left of negative bars
            x_offset = 0.02 if val >= 0 else -0.02

            # Main value with significance
            text = f"{val:.3f}{sig_marker}"
            ax.text(val + x_offset, i,
                    text, va='center',
                    ha='left' if val >= 0 else 'right',
                    fontsize=9, fontweight='bold' if sig_marker else 'normal')

            # Add percentage significant in smaller text below
            if pct_sig > 0:
                ax.text(val + x_offset, i - 0.2,
                        f"({pct_sig:.0f}% sig)",
                        va='center', ha='left' if val >= 0 else 'right',
                        fontsize=7, alpha=0.6, style='italic')

        # Set labels and title
        ax.set_xlabel('Calibrated Effect (SSI_actual - mean(SSI_forced))', fontsize=11)
        ax.set_title(f'Calibrated Effect by Model with Error Bars\n'
                     f'{domain.title()} - {dataset.replace("-", " ").title()}',
                     fontsize=13, fontweight='bold')

        # Add legend for significance levels
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=SIGNIFICANCE_COLORS['high'], alpha=0.8,
                  label='≥50% samples significant'),
            Patch(facecolor=SIGNIFICANCE_COLORS['moderate'], alpha=0.8,
                  label='25-50% significant'),
            Patch(facecolor=SIGNIFICANCE_COLORS['low'], alpha=0.8,
                  label='10-25% significant'),
            Patch(facecolor=SIGNIFICANCE_COLORS['none'], alpha=0.8,
                  label='<10% significant')
        ]
        ax.legend(handles=legend_elements, loc='best', fontsize=8,
                  framealpha=0.9, title='Significance Rate')

        # Add explanatory note
        note_text = ("Error bars: ±1 within-sample SD (from forced choice variance)\n"
                     "Significance: * p<0.05, ** p<0.01, *** p<0.001 (mean z-score)")
        ax.text(0.02, 0.02, note_text, transform=ax.transAxes,
                fontsize=7, alpha=0.6, va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='gray', alpha=0.8))

        # Add light grid for easier reading
        ax.grid(True, axis='x', alpha=0.2, linestyle='--')

        # Adjust layout
        plt.tight_layout()

        # Save figure
        filename = f"calibrated_by_model_{domain}_{dataset}.png"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved: {filename}")

    print(f"\n✓ All plots saved to: {output_dir}")


def create_overall_summary_plot(csv_path, output_dir="plots_with_errorbars", dpi=180):
    """
    Create an overall summary plot showing all domain/dataset combinations.
    """
    output_dir = Path(output_dir)
    df = pd.read_csv(csv_path)

    # Aggregate by domain/dataset
    summary = df.groupby(['domain', 'dataset']).agg({
        'score_calibrated_effect_scorer': ['mean', 'std'],
        'within_sample_std': 'mean',
        'z_score': lambda x: (x.abs() > 1.96).mean() * 100
    }).reset_index()

    summary.columns = ['domain', 'dataset', 'effect_mean', 'effect_std',
                       'within_std_mean', 'pct_significant']

    # Create combined label
    summary['label'] = summary['domain'] + '/' + summary['dataset']

    # Sort by effect
    summary = summary.sort_values('effect_mean')

    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(6, 0.5 * len(summary))))

    y_pos = np.arange(len(summary))

    # Color by domain
    colors = ['#1ABC9C' if d == 'harmfulness' else '#E67E22'
              for d in summary['domain']]

    # Plot with error bars
    ax.barh(y_pos, summary['effect_mean'].values,
            xerr=summary['within_std_mean'].values,
            color=colors, alpha=0.8,
            error_kw={'elinewidth': 1.5, 'capsize': 4, 'alpha': 0.7})

    ax.axvline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.8)

    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(summary['label'], fontsize=9)

    # Annotations
    for i, (_, row) in enumerate(summary.iterrows()):
        val = row['effect_mean']
        pct = row['pct_significant']

        text = f"{val:.3f} ({pct:.0f}%)"
        ax.text(val + (0.02 if val >= 0 else -0.02), i,
                text, va='center',
                ha='left' if val >= 0 else 'right',
                fontsize=8)

    ax.set_xlabel('Mean Calibrated Effect', fontsize=11)
    ax.set_title('Overall Summary: Calibrated Effects by Domain/Dataset',
                 fontsize=13, fontweight='bold')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1ABC9C', alpha=0.8, label='Harmfulness'),
        Patch(facecolor='#E67E22', alpha=0.8, label='Correctness')
    ]
    ax.legend(handles=legend_elements, title='Domain', loc='best')

    ax.grid(True, axis='x', alpha=0.2, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_dir / 'overall_summary.png', dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: overall_summary.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default="samples_with_computed_zscores.csv",
                        help="Path to samples CSV with z-scores")
    parser.add_argument("--out", default="plots_with_errorbars",
                        help="Output directory")
    parser.add_argument("--dpi", type=int, default=180,
                        help="DPI for plots")
    parser.add_argument("--summary", action="store_true",
                        help="Also create overall summary plot")
    args = parser.parse_args()

    print("="*80)
    print("CREATING CALIBRATED EFFECT PLOTS WITH ERROR BARS")
    print("="*80)

    # Create individual plots
    create_calibrated_plots_simple(args.samples, args.out, args.dpi)

    # Create summary if requested
    if args.summary:
        print("\nCreating overall summary plot...")
        create_overall_summary_plot(args.samples, args.out, args.dpi)

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print("\nThe plots now show:")
    print("  • Calibrated effects as horizontal bars")
    print("  • Error bars representing ±1 within-sample standard deviation")
    print("  • Color coding by significance rate")
    print("  • Annotations showing effect size and % significant")


if __name__ == "__main__":
    main()