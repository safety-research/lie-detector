#!/usr/bin/env python3
"""
Self-Sycophancy Data Visualization
Creates vertical bar charts showing ex-ante vs ex-post rating changes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Color schemes
MODEL_COLORS = {
    "gpt": "#3498DB",  # Blue
    "claude": "#E74C3C",  # Red
    "gemini": "#2ECC71",  # Green
    "gemma": "#F39C12",  # Orange
    "other": "#9B59B6"  # Purple
}

DOMAIN_COLORS = {
    "harmfulness": "#16A085",  # Teal
    "correctness": "#D35400"  # Dark Orange
}


def load_data(csv_path="sycophancy_analysis.csv"):
    """Load and prepare the data"""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples")
    print(f"Models: {df['model_short'].unique()}")
    print(f"Domains: {df['domain'].unique()}")
    print(f"Datasets: {df['dataset'].unique()}")
    return df


def create_model_domain_charts(df, output_dir="plots"):
    """Create vertical bar charts for each model/domain combination"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Overall Summary Chart - All Models and Domains
    print("\nCreating overall summary chart...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for idx, domain in enumerate(df['domain'].unique()):
        ax = axes[idx] if idx < 2 else axes[0]
        domain_data = df[df['domain'] == domain]

        # Group by model
        model_stats = domain_data.groupby('model_short').agg({
            'initial_score': 'mean',
            'new_score': 'mean',
            'score_shift': ['mean', 'std', 'count']
        }).reset_index()

        model_stats.columns = ['model', 'initial', 'new', 'shift_mean', 'shift_std', 'count']
        model_stats = model_stats.sort_values('shift_mean', ascending=False)

        x = np.arange(len(model_stats))
        width = 0.35

        # Create bars
        bars1 = ax.bar(x - width / 2, model_stats['initial'], width,
                       label='Ex-Ante (Initial)', color='#3498DB', alpha=0.7,
                       edgecolor='black', linewidth=1)
        bars2 = ax.bar(x + width / 2, model_stats['new'], width,
                       label='Ex-Post (After Choice)', color='#E74C3C', alpha=0.7,
                       edgecolor='black', linewidth=1)

        ax.set_xticks(x)
        ax.set_xticklabels(model_stats['model'], rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('Average Score', fontsize=12)
        ax.set_title(f'{domain.title()} Domain', fontsize=13, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 11)

        # Add shift annotations
        for i, (_, row) in enumerate(model_stats.iterrows()):
            shift = row['shift_mean']
            y_pos = max(row['initial'], row['new']) + 0.3

            # Arrow showing direction
            if abs(shift) > 0.1:
                arrow = '↑' if shift > 0 else '↓'
                color = 'red' if abs(shift) > 1 else 'orange' if abs(shift) > 0.5 else 'gray'
                ax.text(i, y_pos, f'{arrow} {shift:+.2f}', ha='center', fontsize=10,
                        color=color, fontweight='bold')

            # Sample count
            ax.text(i, -0.5, f'n={int(row["count"])}', ha='center', fontsize=8, alpha=0.6)

    plt.suptitle('Self-Sycophancy: Ex-Ante vs Ex-Post Scores by Model', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'overall_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

    # 2. Individual Model Charts
    print("\nCreating individual model charts...")
    for model in df['model_short'].unique():
        model_data = df[df['model_short'] == model]

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        for idx, domain in enumerate(model_data['domain'].unique()):
            ax = axes[idx] if idx < 2 else axes[0]
            domain_model_data = model_data[model_data['domain'] == domain]

            if domain_model_data.empty:
                continue

            # Calculate statistics
            dataset_stats = domain_model_data.groupby('dataset').agg({
                'initial_score': ['mean', 'std'],
                'new_score': ['mean', 'std'],
                'score_shift': ['mean', 'std', 'count']
            }).reset_index()

            dataset_stats.columns = ['dataset', 'initial_mean', 'initial_std',
                                     'new_mean', 'new_std', 'shift_mean', 'shift_std', 'count']

            x = np.arange(len(dataset_stats))
            width = 0.35

            bars1 = ax.bar(x - width / 2, dataset_stats['initial_mean'], width,
                           yerr=dataset_stats['initial_std'], capsize=5,
                           label='Ex-Ante', color='#3498DB', alpha=0.7)
            bars2 = ax.bar(x + width / 2, dataset_stats['new_mean'], width,
                           yerr=dataset_stats['new_std'], capsize=5,
                           label='Ex-Post', color='#E74C3C', alpha=0.7)

            ax.set_xticks(x)
            ax.set_xticklabels(dataset_stats['dataset'], rotation=45, ha='right')
            ax.set_ylabel('Score', fontsize=11)
            ax.set_title(f'{domain.title()} - {model}', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 11)

            # Add annotations
            for i, (_, row) in enumerate(dataset_stats.iterrows()):
                shift = row['shift_mean']
                if abs(shift) > 0.1:
                    y_pos = max(row['initial_mean'], row['new_mean']) + row['new_std'] + 0.3
                    ax.text(i, y_pos, f'Δ={shift:+.2f}', ha='center', fontsize=9,
                            color='red' if abs(shift) > 1 else 'gray')

                ax.text(i, -0.3, f'n={int(row["count"])}', ha='center', fontsize=7, alpha=0.5)

        plt.suptitle(f'Model: {model}', fontsize=13, fontweight='bold')
        plt.tight_layout()
        filename = f'model_{model.replace("/", "_").replace(" ", "_")}.png'
        plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        print(f"  Saved {filename}")

    # 3. Score Shift Distribution
    print("\nCreating shift distribution chart...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, domain in enumerate(df['domain'].unique()):
        ax = axes[idx] if idx < 2 else axes[0]
        domain_data = df[df['domain'] == domain]

        shifts = domain_data['score_shift'].values

        # Create histogram
        n, bins, patches = ax.hist(shifts, bins=20, edgecolor='black', alpha=0.7)

        # Color bars based on shift direction
        for patch, left_edge in zip(patches, bins[:-1]):
            if left_edge > 0.5:
                patch.set_facecolor('#E74C3C')  # Red for positive (sycophantic)
            elif left_edge < -0.5:
                patch.set_facecolor('#3498DB')  # Blue for negative (skeptical)
            else:
                patch.set_facecolor('#95A5A6')  # Gray for neutral

        ax.axvline(0, color='black', linestyle='--', alpha=0.5, linewidth=2)
        ax.set_xlabel('Score Shift (Ex-Post - Ex-Ante)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'{domain.title()} Domain', fontsize=12, fontweight='bold')

        # Add statistics
        mean_shift = shifts.mean()
        median_shift = np.median(shifts)
        std_shift = shifts.std()

        stats_text = f'Mean: {mean_shift:.2f}\nMedian: {median_shift:.2f}\nStd: {std_shift:.2f}\nN: {len(shifts)}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, va='top',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Add percentage labels
        positive = (shifts > 0.5).sum()
        negative = (shifts < -0.5).sum()
        neutral = len(shifts) - positive - negative

        ax.text(0.98, 0.98, f'Sycophantic: {100 * positive / len(shifts):.1f}%\n'
                            f'Skeptical: {100 * negative / len(shifts):.1f}%\n'
                            f'Neutral: {100 * neutral / len(shifts):.1f}%',
                transform=ax.transAxes, va='top', ha='right', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle('Distribution of Score Shifts', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'shift_distributions.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

    # 4. Dataset Comparison
    print("\nCreating dataset comparison chart...")
    fig, ax = plt.subplots(figsize=(14, 8))

    dataset_stats = df.groupby(['dataset', 'domain']).agg({
        'score_shift': ['mean', 'std', 'count']
    }).reset_index()

    dataset_stats.columns = ['dataset', 'domain', 'shift_mean', 'shift_std', 'count']
    dataset_stats = dataset_stats.sort_values('shift_mean', ascending=False)

    # Create grouped bars by domain
    datasets = dataset_stats['dataset'].unique()
    domains = dataset_stats['domain'].unique()

    x = np.arange(len(datasets))
    width = 0.35

    for i, domain in enumerate(domains):
        domain_dataset = dataset_stats[dataset_stats['domain'] == domain]

        # Align with dataset order
        means = []
        stds = []
        for dataset in datasets:
            row = domain_dataset[domain_dataset['dataset'] == dataset]
            if not row.empty:
                means.append(row['shift_mean'].values[0])
                stds.append(row['shift_std'].values[0])
            else:
                means.append(0)
                stds.append(0)

        offset = width * (i - len(domains) / 2 + 0.5)
        color = DOMAIN_COLORS.get(domain, '#95A5A6')

        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=3,
                      label=domain.title(), color=color, alpha=0.7,
                      edgecolor='black', linewidth=1)

        # Add value labels
        for j, (bar, mean) in enumerate(zip(bars, means)):
            if mean != 0:
                ax.text(bar.get_x() + bar.get_width() / 2., mean + 0.1,
                        f'{mean:.2f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Average Score Shift', fontsize=12)
    ax.set_title('Score Shifts by Dataset and Domain', fontsize=14, fontweight='bold')
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'dataset_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

    # 5. Individual Sample Scatter
    print("\nCreating individual sample scatter plot...")
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color by model family
    colors = [MODEL_COLORS.get(family, '#95A5A6') for family in df['model_family']]

    # Create scatter plot
    for model_family in df['model_family'].unique():
        family_data = df[df['model_family'] == model_family]
        ax.scatter(family_data['initial_score'], family_data['new_score'],
                   c=MODEL_COLORS.get(model_family, '#95A5A6'),
                   label=model_family.title(), alpha=0.6, s=50, edgecolor='black', linewidth=0.5)

    # Add y=x line
    ax.plot([0, 10], [0, 10], 'k--', alpha=0.3, label='No change')

    # Add sycophantic/skeptical regions
    ax.fill_between([0, 10], [0, 10], [10, 10], alpha=0.1, color='red', label='Sycophantic region')
    ax.fill_between([0, 10], [0, 0], [0, 10], alpha=0.1, color='blue', label='Skeptical region')

    ax.set_xlabel('Ex-Ante Score (Initial)', fontsize=12)
    ax.set_ylabel('Ex-Post Score (After Choice)', fontsize=12)
    ax.set_title('Individual Sample Score Changes', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 11)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'sample_scatter.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


def print_summary_statistics(df):
    """Print summary statistics"""

    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    print(f"\nTotal samples: {len(df)}")
    print(f"Average initial score: {df['initial_score'].mean():.2f} ± {df['initial_score'].std():.2f}")
    print(f"Average final score: {df['new_score'].mean():.2f} ± {df['new_score'].std():.2f}")
    print(f"Average shift: {df['score_shift'].mean():.3f} ± {df['score_shift'].std():.3f}")

    print("\n=== By Domain ===")
    # Ensure consistent ordering: harmfulness first, then correctness
    domains = df['domain'].unique().tolist()
    domains_ordered = []
    if 'harmfulness' in domains:
        domains_ordered.append('harmfulness')
    if 'correctness' in domains:
        domains_ordered.append('correctness')
    for d in domains:
        if d not in domains_ordered:
            domains_ordered.append(d)

    for domain in domains_ordered:
        if domain not in df['domain'].unique():
            continue
        domain_data = df[df['domain'] == domain]
        print(f"\n{domain.upper()}:")
        print(f"  Samples: {len(domain_data)}")
        print(f"  Mean shift: {domain_data['score_shift'].mean():.3f} ± {domain_data['score_shift'].std():.3f}")

        # Count behaviors
        sycophantic = (domain_data['score_shift'] > 0.5).sum()
        skeptical = (domain_data['score_shift'] < -0.5).sum()
        neutral = len(domain_data) - sycophantic - skeptical

        print(f"  Sycophantic (>0.5): {sycophantic} ({100 * sycophantic / len(domain_data):.1f}%)")
        print(f"  Skeptical (<-0.5): {skeptical} ({100 * skeptical / len(domain_data):.1f}%)")
        print(f"  Neutral: {neutral} ({100 * neutral / len(domain_data):.1f}%)")

    print("\n=== By Model ===")
    for model in df['model_short'].unique():
        model_data = df[df['model_short'] == model]
        print(f"\n{model}:")
        print(f"  Samples: {len(model_data)}")
        print(f"  Mean shift: {model_data['score_shift'].mean():.3f} ± {model_data['score_shift'].std():.3f}")
        print(f"  Initial → Final: {model_data['initial_score'].mean():.2f} → {model_data['new_score'].mean():.2f}")

    print("\n=== Top Sycophantic Samples ===")
    top_syco = df.nlargest(5, 'score_shift')[['model_short', 'dataset', 'initial_score', 'new_score', 'score_shift']]
    print(top_syco.to_string(index=False))

    print("\n=== Top Skeptical Samples ===")
    top_skep = df.nsmallest(5, 'score_shift')[['model_short', 'dataset', 'initial_score', 'new_score', 'score_shift']]
    print(top_skep.to_string(index=False))


def main():
    """Main execution"""

    # Load data
    df = load_data("sycophancy_analysis.csv")

    # Create all charts
    create_model_domain_charts(df)

    # Print statistics
    print_summary_statistics(df)

    print("\n✓ All visualizations complete!")
    print("Check the 'plots' directory for generated charts.")


if __name__ == "__main__":
    main()