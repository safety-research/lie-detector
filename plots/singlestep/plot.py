#!/usr/bin/env python3
"""
Self-Sycophancy Data Visualization
Creates vertical bar charts showing pre-choice vs post-choice rating changes
Modified: All heatmaps show raw counts with column-normalized coloring
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

from matplotlib.colors import LogNorm, Normalize
from scipy import stats

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Color schemes - using new palette
MODEL_COLORS = {
    "gpt": "#01665e",  # Dark teal
    "claude": "#5ab4ac",  # Light teal
    "gemini": "#01665e",  # Dark teal
    "gemma": "#c7eae5",  # Very light teal
    "other": "#5ab4ac"  # Light teal
}

DOMAIN_COLORS = {
    "harmfulness": "#01665e",  # Dark teal
    "correctness": "#5ab4ac"  # Light teal
}


def calculate_95_ci(data):
    """Calculate 95% confidence interval"""
    # Remove NaN values
    clean_data = data[~np.isnan(data)]
    n = len(clean_data)

    if n <= 1:
        return 0

    try:
        sem = stats.sem(clean_data, nan_policy='omit')  # Standard error of the mean
        if np.isnan(sem):
            return 0
        ci = sem * 1.96  # 95% CI
        return ci
    except:
        return 0


def create_transition_heatmap(df, output_dir="plots"):
    """Create heatmaps showing score transitions from pre to post choice"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define score bins (0-10 scale)
    score_bins = np.arange(1, 11)

    # 1. Overall Transition Heatmap
    print("\nCreating overall transition heatmap...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Process overall data - handle NaN values properly
    initial_scores = df['initial_score'].values
    new_scores = df['new_score'].values

    # Create transition matrix
    transition_matrix = np.zeros((10, 10))
    for init, new in zip(initial_scores, new_scores):
        if not np.isnan(init) and not np.isnan(new):
            init_rounded = int(np.round(init))
            new_rounded = int(np.round(new))
            # Ensure values are within bounds
            init_rounded = max(1, min(10, init_rounded))
            new_rounded = max(1, min(10, new_rounded))
            transition_matrix[new_rounded - 1, init_rounded - 1] += 1

    # Normalize by column (what proportion of each initial score goes where)
    transition_matrix_norm = transition_matrix.copy()
    for col in range(10):
        col_sum = transition_matrix[:, col].sum()
        if col_sum > 0:
            transition_matrix_norm[:, col] = transition_matrix[:, col] / col_sum

    # Plot 1: Raw counts (already correct)
    sns.heatmap(transition_matrix, annot=True, fmt='.0f', cmap='YlOrRd',
                ax=axes[0], cbar_kws={'label': 'Count'},
                xticklabels=score_bins, yticklabels=score_bins,
                linewidths=0.5, linecolor='gray')
    axes[0].set_xlabel('Pre-Choice Score', fontsize=12)
    axes[0].set_ylabel('Post-Choice Score', fontsize=12)
    axes[0].set_title('Overall Score Transitions (Counts)', fontsize=14, fontweight='bold')
    axes[0].invert_yaxis()

    # Add diagonal line for reference (no change)
    for i in range(10):
        axes[0].add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                                        edgecolor='blue', lw=2, alpha=0.5))

    # Plot 2: Raw counts with column-normalized coloring
    # Create custom normalization for coloring
    vmin = 0
    vmax = np.max(transition_matrix_norm) if np.max(transition_matrix_norm) > 0 else 1

    # Use normalized matrix for color but raw counts for annotation
    im = axes[1].imshow(transition_matrix_norm, cmap='YlGnBu', aspect='auto',
                        vmin=vmin, vmax=vmax)

    # Add annotations with raw counts
    for i in range(10):
        for j in range(10):
            text = axes[1].text(j, i, f'{int(transition_matrix[i, j])}',
                                ha="center", va="center", color="black")

    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[1])
    cbar.set_label('Normalized Intensity', rotation=270, labelpad=20)

    # Set ticks and labels
    axes[1].set_xticks(np.arange(10))
    axes[1].set_yticks(np.arange(10))
    axes[1].set_xticklabels(score_bins)
    axes[1].set_yticklabels(score_bins)
    axes[1].set_xlabel('Pre-Choice Score', fontsize=12)
    axes[1].set_ylabel('Post-Choice Score', fontsize=12)
    axes[1].set_title('Score Transitions (Counts, Column-Normalized Color)', fontsize=14, fontweight='bold')

    # Add grid
    for edge, spine in axes[1].spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color('gray')

    # Add diagonal line for reference
    for i in range(10):
        axes[1].add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                        edgecolor='blue', lw=2, alpha=0.5))

    # Plot 3: Difference from diagonal with raw counts
    # Create a matrix showing net movement with raw counts
    net_movement_counts = np.zeros((10, 10))
    net_movement_norm = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            if i > j:  # Above diagonal = increase (sycophantic)
                net_movement_counts[i, j] = transition_matrix[i, j]
                net_movement_norm[i, j] = transition_matrix_norm[i, j]
            elif i < j:  # Below diagonal = decrease (skeptical)
                net_movement_counts[i, j] = -transition_matrix[i, j]
                net_movement_norm[i, j] = -transition_matrix_norm[i, j]
            # Diagonal (i == j) remains 0 to show no change

    # Custom colormap - red for skeptical, green for sycophantic
    # Use normalized values for color, raw counts for annotation
    vmax_abs = np.max(np.abs(net_movement_norm))
    sns.heatmap(net_movement_norm, annot=net_movement_counts, fmt='.0f', cmap='RdYlGn',
                ax=axes[2], cbar_kws={'label': 'Direction (- skeptical, + sycophantic)'},
                xticklabels=score_bins, yticklabels=score_bins,
                linewidths=0.5, linecolor='gray', center=0, vmin=-vmax_abs, vmax=vmax_abs)
    axes[2].set_xlabel('Pre-Choice Score', fontsize=12)
    axes[2].set_ylabel('Post-Choice Score', fontsize=12)
    axes[2].set_title('Directional Movement Pattern (Raw Counts)', fontsize=14, fontweight='bold')
    axes[2].invert_yaxis()

    # Add diagonal line
    for i in range(10):
        axes[2].add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                                        edgecolor='black', lw=2, alpha=0.7))

    plt.suptitle('Score Transition Analysis', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / 'transition_heatmap_overall.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

    # 2. Domain-specific heatmaps
    print("\nCreating domain-specific transition heatmaps...")
    domains = df['domain'].unique()

    fig, axes = plt.subplots(2, len(domains), figsize=(8 * len(domains), 14))

    for idx, domain in enumerate(domains):
        domain_data = df[df['domain'] == domain]

        domain_data = domain_data[(domain_data['initial_score'] != 0) & (domain_data['new_score'] != 0)]

        # Process domain data - handle NaN values properly
        initial_scores = domain_data['initial_score'].values
        new_scores = domain_data['new_score'].values

        # Create transition matrix
        transition_matrix = np.zeros((10, 10))
        for init, new in zip(initial_scores, new_scores):
            if not np.isnan(init) and not np.isnan(new):
                init_rounded = int(np.round(init))
                new_rounded = int(np.round(new))
                # Ensure values are within bounds
                init_rounded = max(1, min(10, init_rounded))
                new_rounded = max(1, min(10, new_rounded))
                transition_matrix[new_rounded - 1, init_rounded - 1] += 1

        # Normalize
        transition_matrix_norm = transition_matrix.copy()
        for col in range(10):
            col_sum = transition_matrix[:, col].sum()
            if col_sum > 0:
                transition_matrix_norm[:, col] = transition_matrix[:, col] / col_sum

        # Plot counts (already correct with LogNorm)
        ax1 = axes[0, idx] if len(domains) > 1 else axes[0]
        sns.heatmap(transition_matrix, annot=True, fmt='.0f', cmap='YlOrRd',
                    ax=ax1, cbar_kws={'label': 'Count'},
                    norm=LogNorm(vmin=0.1, vmax=transition_matrix.max()) if transition_matrix.max() > 0 else None,
                    xticklabels=score_bins, yticklabels=score_bins,
                    linewidths=0.5, linecolor='gray')
        ax1.set_xlabel('Pre-Choice Score', fontsize=11)
        ax1.set_ylabel('Post-Choice Score', fontsize=11)
        domain_title = 'Harmfulness MCQ' if domain == 'harmfulness' else 'Correctness MCQ'
        ax1.set_title(f'{domain_title} - Counts', fontsize=12, fontweight='bold')
        ax1.invert_yaxis()

        # Add diagonal
        for i in range(10):
            ax1.add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                                        edgecolor='blue', lw=1.5, alpha=0.5))

        # Plot raw counts with normalized coloring
        ax2 = axes[1, idx] if len(domains) > 1 else axes[1]

        # Use normalized matrix for color but raw counts for annotation
        vmin = 0
        vmax = np.max(transition_matrix_norm) if np.max(transition_matrix_norm) > 0 else 1

        im = ax2.imshow(transition_matrix_norm, cmap='YlGnBu', aspect='auto',
                        vmin=vmin, vmax=vmax)

        # Add annotations with raw counts
        for i in range(10):
            for j in range(10):
                text = ax2.text(j, i, f'{int(transition_matrix[i, j])}',
                                ha="center", va="center", color="black")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Normalized Intensity', rotation=270, labelpad=20)

        # Set ticks and labels
        ax2.set_xticks(np.arange(10))
        ax2.set_yticks(np.arange(10))
        ax2.set_xticklabels(score_bins)
        ax2.set_yticklabels(score_bins)
        ax2.set_xlabel('Pre-Choice Score', fontsize=11)
        ax2.set_ylabel('Post-Choice Score', fontsize=11)
        ax2.set_title(f'{domain_title} - Counts (Column-Normalized Color)', fontsize=12, fontweight='bold')

        # Add grid
        for edge, spine in ax2.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_color('gray')

        # Add diagonal
        for i in range(10):
            ax2.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                        edgecolor='blue', lw=1.5, alpha=0.5))

    plt.suptitle('Domain-Specific Score Transitions', fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(output_dir / 'transition_heatmap_domains.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

    # 3. Model-specific transition patterns (for top models by sample size)
    print("\nCreating model-specific transition heatmaps...")

    # Get top 6 models by sample count
    model_counts = df['model_short'].value_counts()
    top_models = model_counts.head(9).index.tolist()

    # Create separate figures for each domain
    for domain in df['domain'].unique():
        domain_data = df[df['domain'] == domain]

        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, model in enumerate(top_models):
            if idx >= 9:
                break

            model_domain_data = domain_data[domain_data['model_short'] == model]
            model_domain_data = model_domain_data[
                (model_domain_data['initial_score'] != 0) & (model_domain_data['new_score'] != 0)]

            if model_domain_data.empty:
                axes[idx].axis('off')
                continue

            # Process model data - handle NaN values properly
            initial_scores = model_domain_data['initial_score'].values
            new_scores = model_domain_data['new_score'].values

            # Create transition matrix
            transition_matrix = np.zeros((10, 10))
            for init, new in zip(initial_scores, new_scores):
                if not np.isnan(init) and not np.isnan(new):
                    init_rounded = int(np.round(init))
                    new_rounded = int(np.round(new))
                    # Ensure values are within bounds
                    init_rounded = max(1, min(10, init_rounded))
                    new_rounded = max(1, min(10, new_rounded))
                    transition_matrix[new_rounded - 1, init_rounded - 1] += 1

            # Normalize
            transition_matrix_norm = transition_matrix.copy()
            for col in range(10):
                col_sum = transition_matrix[:, col].sum()
                if col_sum > 0:
                    transition_matrix_norm[:, col] = transition_matrix[:, col] / col_sum

            # Plot raw counts with normalized coloring
            ax = axes[idx]

            # Use normalized matrix for color but raw counts for annotation
            vmin = 0
            vmax = np.max(transition_matrix_norm) if np.max(transition_matrix_norm) > 0 else 1

            im = ax.imshow(transition_matrix_norm, cmap='YlGnBu', aspect='auto',
                           vmin=vmin, vmax=vmax)

            # Add annotations with raw counts
            for i in range(10):
                for j in range(10):
                    count = int(transition_matrix[i, j])
                    if count > 0:  # Only show non-zero counts
                        text = ax.text(j, i, f'{count}',
                                       ha="center", va="center", color="black", fontsize=8)

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.ax.tick_params(labelsize=8)

            # Set ticks and labels
            ax.set_xticks(np.arange(10))
            ax.set_yticks(np.arange(10))
            ax.set_xticklabels(score_bins, fontsize=8)
            ax.set_yticklabels(score_bins, fontsize=8)
            ax.set_xlabel('Pre-Choice Score', fontsize=10)
            ax.set_ylabel('Post-Choice Score', fontsize=10)
            ax.set_title(f'{model}\n(n={len(model_domain_data)})', fontsize=11, fontweight='bold')

            # Add grid
            for edge, spine in ax.spines.items():
                spine.set_visible(True)
                spine.set_linewidth(0.5)
                spine.set_color('gray')

            # Add diagonal
            for i in range(10):
                ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor='blue', lw=1.5, alpha=0.5))

        plt.suptitle(f'Model-Specific Transition Patterns (Top 9 Models) ({domain})',
                     fontsize=14, fontweight='bold', y=0.99)
        plt.tight_layout()
        plt.savefig(output_dir / f'transition_heatmap_models_{domain}.png', dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()

    # 4. Summary statistics from transitions
    print("\n" + "=" * 60)
    print("TRANSITION ANALYSIS SUMMARY")
    print("=" * 60)

    # Calculate overall statistics - handle NaN values
    all_transitions = []
    for init, new in zip(df['initial_score'].values, df['new_score'].values):
        if not np.isnan(init) and not np.isnan(new):
            all_transitions.append(new - init)

    all_transitions = np.array(all_transitions)

    print(f"\nTotal transitions analyzed: {len(all_transitions)}")
    print(f"Mean change: {np.mean(all_transitions):.2f}")
    print(f"Median change: {np.median(all_transitions):.2f}")
    print(f"Std deviation: {np.std(all_transitions):.2f}")

    # Categorize transitions
    no_change = np.sum(np.abs(all_transitions) < 0.5)
    small_change = np.sum((np.abs(all_transitions) >= 0.5) & (np.abs(all_transitions) < 2))
    medium_change = np.sum((np.abs(all_transitions) >= 2) & (np.abs(all_transitions) < 4))
    large_change = np.sum(np.abs(all_transitions) >= 4)

    print(f"\nTransition magnitudes:")
    print(f"  No change (±0.5): {no_change} ({100 * no_change / len(all_transitions):.1f}%)")
    print(f"  Small (±0.5-2): {small_change} ({100 * small_change / len(all_transitions):.1f}%)")
    print(f"  Medium (±2-4): {medium_change} ({100 * medium_change / len(all_transitions):.1f}%)")
    print(f"  Large (±4+): {large_change} ({100 * large_change / len(all_transitions):.1f}%)")

    # Direction analysis
    increases = np.sum(all_transitions > 0.5)
    decreases = np.sum(all_transitions < -0.5)

    print(f"\nDirection:")
    print(f"  Increases (sycophantic): {increases} ({100 * increases / len(all_transitions):.1f}%)")
    print(f"  Decreases (skeptical): {decreases} ({100 * decreases / len(all_transitions):.1f}%)")
    print(f"  Stable: {no_change} ({100 * no_change / len(all_transitions):.1f}%)")

    # 5. Create a focused transition flow diagram with raw counts
    print("\nCreating transition flow diagram...")
    fig, ax = plt.subplots(figsize=(14, 10))

    # Calculate aggregated transitions for cleaner visualization
    initial_scores = df['initial_score'].values
    new_scores = df['new_score'].values

    # Group scores into bins for clearer flow
    score_groups = {
        'Low (0-3)': (0, 3),
        'Medium-Low (4-5)': (4, 5),
        'Medium-High (6-7)': (6, 7),
        'High (8-10)': (8, 10)
    }

    # Create grouped transition matrix
    group_transitions = np.zeros((len(score_groups), len(score_groups)))
    group_labels = list(score_groups.keys())

    for init, new in zip(initial_scores, new_scores):
        if not np.isnan(init) and not np.isnan(new):
            init_group = None
            new_group = None

            for i, (label, (low, high)) in enumerate(score_groups.items()):
                if low <= init <= high:
                    init_group = i
                if low <= new <= high:
                    new_group = i

            if init_group is not None and new_group is not None:
                group_transitions[new_group, init_group] += 1

    # Normalize for coloring only
    group_transitions_norm = group_transitions.copy()
    for col in range(len(score_groups)):
        col_sum = group_transitions[:, col].sum()
        if col_sum > 0:
            group_transitions_norm[:, col] = group_transitions[:, col] / col_sum

    # Create heatmap with raw counts as annotations but normalized coloring
    vmin = 0
    vmax = np.max(group_transitions_norm) if np.max(group_transitions_norm) > 0 else 1

    im = ax.imshow(group_transitions_norm, cmap='coolwarm', aspect='auto',
                   vmin=vmin, vmax=vmax, origin='lower')

    # Add annotations with raw counts
    for i in range(len(score_groups)):
        for j in range(len(score_groups)):
            count = int(group_transitions[i, j])
            text = ax.text(j, i, f'{count}',
                           ha="center", va="center", color="black", fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Intensity', rotation=270, labelpad=20)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(score_groups)))
    ax.set_yticks(np.arange(len(score_groups)))
    ax.set_xticklabels(group_labels)
    ax.set_yticklabels(group_labels)
    ax.set_xlabel('Pre-Choice Score Group', fontsize=12, fontweight='bold')
    ax.set_ylabel('Post-Choice Score Group', fontsize=12, fontweight='bold')
    ax.set_title('Simplified Score Transition Flow (Raw Counts)\n(Where does the mass go?)',
                 fontsize=14, fontweight='bold')
    ax.invert_yaxis()

    # Highlight diagonal (no group change)
    for i in range(len(score_groups)):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor='green', lw=3, alpha=0.7))

    # Add text annotations for interpretation
    ax.text(0.5, -0.15,
            "Green boxes = Same group (minimal change) | Above diagonal = Score increase | Below diagonal = Score decrease",
            transform=ax.transAxes, ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    plt.savefig(output_dir / 'transition_flow_simplified.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


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

    # 1. Overall Summary Charts - Separate for each domain
    print("\nCreating overall summary charts...")

    for domain in df['domain'].unique():
        # Use wider figure for long model names
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))

        domain_data = df[df['domain'] == domain]

        # Group by model and calculate statistics
        model_groups = []
        for model in domain_data['model_short'].unique():
            model_data = domain_data[domain_data['model_short'] == model]

            initial_scores = model_data['initial_score'].values
            new_scores = model_data['new_score'].values

            model_groups.append({
                'model': model,
                'initial_mean': np.nanmean(initial_scores),
                'new_mean': np.nanmean(new_scores),
                'initial_ci': calculate_95_ci(initial_scores),
                'new_ci': calculate_95_ci(new_scores),
                'shift_mean': np.nanmean(model_data['score_shift'].values),
                'count': len(model_data)
            })

        model_stats = pd.DataFrame(model_groups)
        # Sort alphabetically by model name
        model_stats = model_stats.sort_values('model')

        x = np.arange(len(model_stats))
        width = 0.35

        # Create bars with 95% CI error bars and borders using new color palette
        # CHANGED: Updated labels for legend
        bars1 = ax.bar(x - width / 2, model_stats['initial_mean'], width,
                       yerr=model_stats['initial_ci'], capsize=5,
                       label='Baseline', color='#01665e', alpha=0.9,
                       edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width / 2, model_stats['new_mean'], width,
                       yerr=model_stats['new_ci'], capsize=5,
                       label='Same Turn Self Attributed', color='#5ab4ac', alpha=0.9,
                       edgecolor='black', linewidth=1.5)

        ax.set_xticks(x)
        # CHANGED: Rotate labels to 45 degrees and increase fontsize
        ax.set_xticklabels(model_stats['model'], rotation=45, ha='right', fontweight='bold', fontsize=13)
        ax.set_ylabel('Average Score', fontsize=12)

        # Set title based on domain with N reference
        domain_title = 'Harmfulness MCQ' if domain == 'harmfulness' else 'Correctness MCQ'
        avg_n = int(round(model_stats['count'].mean()))
        ax.set_title(f'{domain_title}\n(n ≈ {avg_n} per model)', fontsize=14, fontweight='bold')

        # REMOVED: Red average line code block was here

        # Move legend below the chart, closer to reduce whitespace
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=2, frameon=True,fontsize=13,borderpad=1.0, columnspacing=2.0)
        ax.grid(True, alpha=0.3, axis='y')

        # Adaptive Y-axis limits
        all_values = list(model_stats['initial_mean']) + list(model_stats['new_mean'])
        all_errors = list(model_stats['initial_ci']) + list(model_stats['new_ci'])
        max_val = max(all_values) + max(all_errors)
        min_val = min(all_values) - max(all_errors)
        y_margin = (max_val - min_val) * 0.15  # 15% margin
        ax.set_ylim(max(0, min_val - y_margin), min(11, max_val + y_margin))

        # Add shift annotations with adjusted position for adaptive Y-axis
        for i, (_, row) in enumerate(model_stats.iterrows()):
            shift = row['shift_mean']
            # Calculate position based on actual data range
            y_pos = max(row['initial_mean'] + row['initial_ci'],
                        row['new_mean'] + row['new_ci']) + y_margin * 0.3

            # Arrow showing direction
            if abs(shift) > 0.1:
                arrow = '↑' if shift > 0 else '↓'
                color = '#01665e' if abs(shift) > 1 else '#5ab4ac' if abs(shift) > 0.5 else 'gray'
                ax.text(i, y_pos, f'{arrow} {shift:+.2f}', ha='center', fontsize=10,
                        color=color, fontweight='bold')

        plt.tight_layout()
        filename = f'overall_{domain}_comparison.png'
        plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        print(f"  Saved {filename}")

    # 2. Individual Model Charts
    print("\nCreating individual model charts...")
    for model in sorted(df['model_short'].unique()):  # Sort models alphabetically
        model_data = df[df['model_short'] == model]

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        for idx, domain in enumerate(model_data['domain'].unique()):
            ax = axes[idx] if idx < 2 else axes[0]
            domain_model_data = model_data[model_data['domain'] == domain]

            if domain_model_data.empty:
                continue

            # Calculate statistics with 95% CI
            dataset_groups = []
            for dataset in domain_model_data['dataset'].unique():
                dataset_data = domain_model_data[domain_model_data['dataset'] == dataset]

                initial_scores = dataset_data['initial_score'].values
                new_scores = dataset_data['new_score'].values

                dataset_groups.append({
                    'dataset': dataset,
                    'initial_mean': np.nanmean(initial_scores),
                    'initial_ci': calculate_95_ci(initial_scores),
                    'new_mean': np.nanmean(new_scores),
                    'new_ci': calculate_95_ci(new_scores),
                    'shift_mean': np.nanmean(dataset_data['score_shift'].values),
                    'count': len(dataset_data)
                })

            dataset_stats = pd.DataFrame(dataset_groups)
            # Sort datasets alphabetically
            dataset_stats = dataset_stats.sort_values('dataset')

            x = np.arange(len(dataset_stats))
            width = 0.35

            # CHANGED: Updated labels for legend
            bars1 = ax.bar(x - width / 2, dataset_stats['initial_mean'], width,
                           yerr=dataset_stats['initial_ci'], capsize=5,
                           label='Baseline', color='#01665e', alpha=0.9,
                           edgecolor='black', linewidth=1.5)
            bars2 = ax.bar(x + width / 2, dataset_stats['new_mean'], width,
                           yerr=dataset_stats['new_ci'], capsize=5,
                           label='Same Turn Self Attributed', color='#5ab4ac', alpha=0.9,
                           edgecolor='black', linewidth=1.5)

            ax.set_xticks(x)
            ax.set_xticklabels(dataset_stats['dataset'], rotation=45, fontweight='bold', ha='right')
            ax.set_ylabel('Score', fontsize=11)

            # Update domain titles with N at top
            domain_title = 'Harmfulness MCQ' if domain == 'harmfulness' else 'Correctness MCQ'
            total_samples = int(dataset_stats['count'].sum())
            ax.set_title(f'{domain_title} - {model}\n(n = {total_samples})', fontsize=12, fontweight='bold')

            # Move legend below the chart, closer to reduce whitespace
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), fontsize=13, ncol=2, frameon=True, borderpad=1.0, columnspacing=2.0)
            ax.grid(True, alpha=0.3, axis='y')

            # Adaptive Y-axis limits
            all_values = list(dataset_stats['initial_mean']) + list(dataset_stats['new_mean'])
            all_errors = list(dataset_stats['initial_ci']) + list(dataset_stats['new_ci'])
            if all_values:
                max_val = max(all_values) + max(all_errors)
                min_val = min(all_values) - max(all_errors)
                y_margin = (max_val - min_val) * 0.15  # 15% margin
                ax.set_ylim(max(0, min_val - y_margin), min(11, max_val + y_margin))
            else:
                ax.set_ylim(0, 11)

            # Add annotations with adjusted position for adaptive Y-axis
            for i, (_, row) in enumerate(dataset_stats.iterrows()):
                shift = row['shift_mean']
                if abs(shift) > 0.1:
                    # Calculate position based on actual data range
                    y_pos = max(row['initial_mean'] + row['initial_ci'],
                                row['new_mean'] + row['new_ci']) + y_margin * 0.2
                    ax.text(i, y_pos, f'Δ={shift:+.2f}', ha='center', fontsize=9,
                            color='#01665e' if abs(shift) > 1 else 'gray')

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
        # Remove NaN values for histogram
        shifts = shifts[~np.isnan(shifts)]

        # Create histogram
        n, bins, patches = ax.hist(shifts, bins=20, edgecolor='black', linewidth=1.5, alpha=0.9)

        # Color bars based on shift direction using new palette
        for patch, left_edge in zip(patches, bins[:-1]):
            if left_edge > 0.5:
                patch.set_facecolor('#5ab4ac')  # Light teal for positive (sycophantic)
            elif left_edge < -0.5:
                patch.set_facecolor('#01665e')  # Dark teal for negative (skeptical)
            else:
                patch.set_facecolor('#c7eae5')  # Very light teal for neutral

        ax.axvline(0, color='black', linestyle='--', alpha=0.5, linewidth=2)
        ax.set_xlabel('Score Shift (Post-Choice - Pre-Choice)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)

        # Update domain titles
        domain_title = 'Harmfulness MCQ' if domain == 'harmfulness' else 'Correctness MCQ'
        ax.set_title(domain_title, fontsize=12, fontweight='bold')

        # Add statistics
        mean_shift = np.nanmean(shifts)
        median_shift = np.nanmedian(shifts)
        std_shift = np.nanstd(shifts)
        ci_95 = calculate_95_ci(shifts)

        stats_text = f'Mean: {mean_shift:.2f} ± {ci_95:.2f}\nMedian: {median_shift:.2f}\nStd: {std_shift:.2f}\nN: {len(shifts)}'
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

    # Calculate statistics with 95% CI
    dataset_groups = []
    for dataset in df['dataset'].unique():
        for domain in df['domain'].unique():
            domain_dataset_data = df[(df['dataset'] == dataset) & (df['domain'] == domain)]
            if not domain_dataset_data.empty:
                shifts = domain_dataset_data['score_shift'].values
                dataset_groups.append({
                    'dataset': dataset,
                    'domain': domain,
                    'shift_mean': np.nanmean(shifts),
                    'shift_ci': calculate_95_ci(shifts),
                    'count': len(shifts)
                })

    dataset_stats = pd.DataFrame(dataset_groups)
    # Sort by dataset name alphabetically
    datasets = sorted(dataset_stats['dataset'].unique())
    domains = dataset_stats['domain'].unique()

    x = np.arange(len(datasets))
    width = 0.35

    for i, domain in enumerate(domains):
        domain_dataset = dataset_stats[dataset_stats['domain'] == domain]

        # Align with dataset order
        means = []
        cis = []
        for dataset in datasets:
            row = domain_dataset[domain_dataset['dataset'] == dataset]
            if not row.empty:
                means.append(row['shift_mean'].values[0])
                cis.append(row['shift_ci'].values[0])
            else:
                means.append(0)
                cis.append(0)

        offset = width * (i - len(domains) / 2 + 0.5)
        color = DOMAIN_COLORS.get(domain, '#c7eae5')

        # Update domain labels
        domain_label = 'Harmfulness MCQ' if domain == 'harmfulness' else 'Correctness MCQ'

        bars = ax.bar(x + offset, means, width, yerr=cis, capsize=3,
                      label=domain_label, color=color, alpha=0.9,
                      edgecolor='black', linewidth=1.5)

        # Add value labels with proper positioning
        for j, (bar, mean, ci) in enumerate(zip(bars, means, cis)):
            if mean != 0:
                # Position above or below bar depending on value
                if mean > 0:
                    y_pos = mean + ci + 0.05
                else:
                    y_pos = mean - ci - 0.05
                ax.text(bar.get_x() + bar.get_width() / 2., y_pos,
                        f'{mean:.2f}', ha='center', va='bottom' if mean > 0 else 'top',
                        fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right', fontweight='bold', fontsize=10)
    ax.set_ylabel('Average Score Shift', fontsize=12)
    ax.set_title('Score Shifts by Dataset and Domain', fontsize=14, fontweight='bold')
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)

    # Move legend below the chart, closer to reduce whitespace
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=13, frameon=True, borderpad=1.0, columnspacing=2.0)

    ax.grid(True, alpha=0.3)#, axis='y')

    # Adaptive Y-axis limits for shifts
    all_means = []
    all_cis = []
    for _, row in dataset_stats.iterrows():
        all_means.append(row['shift_mean'])
        all_cis.append(row['shift_ci'])

    if all_means:
        max_val = max(all_means) + max(all_cis)
        min_val = min(all_means) - max(all_cis)
        y_margin = max(0.5, (max_val - min_val) * 0.15)  # At least 0.5 margin for shifts
        ax.set_ylim(min_val - y_margin, max_val + y_margin)

    plt.tight_layout()
    plt.savefig(output_dir / 'dataset_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

    # 5. Individual Sample Scatter
    print("\nCreating individual sample scatter plot...")
    fig, ax = plt.subplots(figsize=(12, 8))

    # Sort model families alphabetically for consistent ordering
    model_families = sorted(df['model_family'].unique())

    # Create scatter plot
    for model_family in model_families:
        family_data = df[df['model_family'] == model_family]
        ax.scatter(family_data['initial_score'], family_data['new_score'],
                   c=MODEL_COLORS.get(model_family, '#c7eae5'),
                   label=model_family.title(), alpha=0.7, s=50,
                   edgecolor='black', linewidth=0.5)

    # Add y=x line
    ax.plot([0, 10], [0, 10], 'k--', alpha=0.3, label='No change')

    # Add sycophantic/skeptical regions using new palette colors
    ax.fill_between([0, 10], [0, 10], [10, 10], alpha=0.1, color='#5ab4ac', label='Sycophantic region')
    ax.fill_between([0, 10], [0, 0], [0, 10], alpha=0.1, color='#01665e', label='Skeptical region')

    ax.set_xlabel('Pre-Choice Score', fontsize=12)
    ax.set_ylabel('Post-Choice Score', fontsize=12)
    ax.set_title('Individual Sample Score Changes', fontsize=14, fontweight='bold')

    # Adaptive axis limits based on actual data range
    all_initial = df['initial_score'].dropna()
    all_new = df['new_score'].dropna()
    if len(all_initial) > 0 and len(all_new) > 0:
        min_score = min(all_initial.min(), all_new.min())
        max_score = max(all_initial.max(), all_new.max())
        margin = (max_score - min_score) * 0.05
        ax.set_xlim(max(0, min_score - margin), min(11, max_score + margin))
        ax.set_ylim(max(0, min_score - margin), min(11, max_score + margin))
    else:
        ax.set_xlim(0, 11)
        ax.set_ylim(0, 11)

    # Move legend to bottom with multiple columns, closer to reduce whitespace
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=13, frameon=True, borderpad=1.0, columnspacing=2.0)
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

        # Calculate 95% CI for domain shifts
        shifts = domain_data['score_shift'].values
        ci_95 = calculate_95_ci(shifts)
        mean_shift = np.nanmean(shifts)

        print(f"\n{domain.upper()}:")
        print(f"  Samples: {len(domain_data)}")
        print(f"  Mean shift: {mean_shift:.3f} ± {ci_95:.3f} (95% CI)")

        # Count behaviors - handle NaN values
        shifts_clean = shifts[~np.isnan(shifts)]
        sycophantic = (shifts_clean > 0.5).sum()
        skeptical = (shifts_clean < -0.5).sum()
        neutral = len(shifts_clean) - sycophantic - skeptical

        print(f"  Sycophantic (>0.5): {sycophantic} ({100 * sycophantic / len(shifts_clean):.1f}%)")
        print(f"  Skeptical (<-0.5): {skeptical} ({100 * skeptical / len(shifts_clean):.1f}%)")
        print(f"  Neutral: {neutral} ({100 * neutral / len(shifts_clean):.1f}%)")

    print("\n=== By Model ===")
    for model in sorted(df['model_short'].unique()):  # Sort models alphabetically
        model_data = df[df['model_short'] == model]

        # Calculate 95% CI for model shifts
        shifts = model_data['score_shift'].values
        ci_95 = calculate_95_ci(shifts)
        mean_shift = np.nanmean(shifts)

        print(f"\n{model}:")
        print(f"  Samples: {len(model_data)}")
        print(f"  Mean shift: {mean_shift:.3f} ± {ci_95:.3f} (95% CI)")
        print(f"  Initial → Final: {model_data['initial_score'].mean():.2f} → {model_data['new_score'].mean():.2f}")

    print("\n=== Top Sycophantic Samples ===")
    # Filter out NaN values before getting top samples
    df_clean = df.dropna(subset=['score_shift'])
    top_syco = df_clean.nlargest(5, 'score_shift')[
        ['model_short', 'dataset', 'initial_score', 'new_score', 'score_shift']]
    print(top_syco.to_string(index=False))

    print("\n=== Top Skeptical Samples ===")
    top_skep = df_clean.nsmallest(5, 'score_shift')[
        ['model_short', 'dataset', 'initial_score', 'new_score', 'score_shift']]
    print(top_skep.to_string(index=False))


def main():
    """Main execution"""

    # Load data
    df = load_data("sycophancy_analysis.csv")

    # Create all charts
    create_model_domain_charts(df, output_dir="plots2")
    create_transition_heatmap(df, output_dir="plots2")
    # Print statistics
    print_summary_statistics(df)

    print("\n✓ All visualizations complete!")
    print("Check the 'plots2' directory for generated charts.")


if __name__ == "__main__":
    main()