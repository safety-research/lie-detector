#!/usr/bin/env python3
"""
Computer Use Harmfulness Score Transition Heatmap
Shows how harmfulness assessments shift from baseline to followup/continuation
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from inspect_ai.log import read_eval_log
from typing import Dict, List, Tuple
import seaborn as sns
import warnings

from matplotlib.colors import LogNorm

warnings.filterwarnings('ignore')


def extract_individual_transitions(log_path: str) -> Dict:
    """
    Extract individual score transitions from baseline to followup/continuation.

    Returns:
    --------
    dict : Contains transition pairs and model info
    """
    log = read_eval_log(log_path)
    model_name = log.eval.model

    # Clean model name
    if '/' in model_name:
        model_name = model_name.split('/')[-1]

    transitions = {
        'model_name': model_name,
        'baseline_to_followup': [],
        'baseline_to_continuation': [],
        'true_risks': []
    }

    for sample in log.samples:
        metadata = sample.metadata

        # Get true risk
        true_risk = None
        if hasattr(sample, 'target') and sample.target:
            try:
                true_risk = float(sample.target)
            except:
                pass
        if true_risk is None and 'true_risk_level' in metadata:
            true_risk = float(metadata['true_risk_level'])

        if true_risk is None:
            continue

        # Get baseline score
        baseline_score = None
        if 'baseline_mean' in metadata:
            baseline_score = metadata['baseline_mean']
        elif 'baseline_risks' in metadata:
            baseline_score = np.mean(metadata['baseline_risks'])

        # Get followup score
        followup_score = None
        if 'prefilled_mean' in metadata:
            followup_score = metadata['prefilled_mean']
        elif 'prefilled_risk_score' in metadata:
            followup_score = metadata['prefilled_risk_score']
        elif 'prefilled_risks' in metadata:
            followup_score = np.mean(metadata['prefilled_risks'])

        # Get continuation score
        continuation_score = None
        if 'continuation_risk_score' in metadata:
            continuation_score = metadata['continuation_risk_score']

        # Store transitions
        if baseline_score is not None and followup_score is not None:
            transitions['baseline_to_followup'].append((baseline_score, followup_score))
            transitions['true_risks'].append(true_risk)

        if baseline_score is not None and continuation_score is not None:
            transitions['baseline_to_continuation'].append((baseline_score, continuation_score))

    return transitions


def create_transition_heatmaps(log_directory: str, output_dir: str = "plots"):
    """
    Create transition heatmaps for all models in the directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all logs
    log_dir = Path(log_directory)
    log_files = list(log_dir.glob("*.eval"))
    if not log_files:
        log_files = list(log_dir.glob("*.json"))

    print(f"Found {len(log_files)} log files")

    # Extract transitions for all models
    all_transitions = {}
    for log_file in log_files:
        print(f"Processing: {log_file.name}")
        try:
            transitions = extract_individual_transitions(str(log_file))
            model_name = transitions['model_name']
            all_transitions[model_name] = transitions
            print(
                f"  ✓ Extracted {len(transitions['baseline_to_followup'])} followup and {len(transitions['baseline_to_continuation'])} continuation transitions for {model_name}")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    if not all_transitions:
        print("No data to plot!")
        return

    # Define common settings
    bins = np.arange(0, 110, 10)  # 0, 10, 20, ..., 100
    bin_labels = ['0-10', '10-20', '20-30', '30-40', '40-50',
                  '50-60', '60-70', '70-80', '80-90', '90-100']

    # 1. Create combined FOLLOWUP transition heatmap for all models
    print("\nCreating combined FOLLOWUP transition heatmap...")
    create_combined_heatmap(all_transitions, 'baseline_to_followup',
                            'Same Turn Self Attributed', bin_labels, output_dir)

    # 2. Create combined CONTINUATION transition heatmap for all models
    print("\nCreating combined CONTINUATION transition heatmap...")
    create_combined_heatmap(all_transitions, 'baseline_to_continuation',
                            'Previous Turn Self Attributed', bin_labels, output_dir)

    # 3. Create individual model heatmaps for FOLLOWUP
    print("\nCreating individual model FOLLOWUP transition heatmaps...")
    create_individual_model_heatmaps(all_transitions, 'baseline_to_followup',
                                     'Same Turn Self Attributed', bin_labels, output_dir)

    # 4. Create individual model heatmaps for CONTINUATION
    print("\nCreating individual model CONTINUATION transition heatmaps...")
    create_individual_model_heatmaps(all_transitions, 'baseline_to_continuation',
                                     'Previous Turn Self Attributed', bin_labels, output_dir)

    # 5. Create comparison heatmap (Followup vs Continuation)
    print("\nCreating comparison heatmap (Followup vs Continuation)...")
    create_comparison_heatmap(all_transitions, bin_labels, output_dir)

    # 6. Print transition statistics for both types
    print_transition_statistics(all_transitions)


def create_combined_heatmap(all_transitions, transition_type, transition_name, bin_labels, output_dir):
    """Create combined transition heatmap for all models."""

    # Combine all transitions
    all_transitions_list = []
    for model_data in all_transitions.values():
        all_transitions_list.extend(model_data[transition_type])

    if not all_transitions_list:
        print(f"  No {transition_name} transitions found!")
        return

    # Create transition matrix (10x10 for 0-100 in bins of 10)
    transition_matrix = np.zeros((10, 10))
    for baseline, target in all_transitions_list:
        baseline_bin = min(int(baseline / 10), 9)
        target_bin = min(int(target / 10), 9)
        transition_matrix[target_bin, baseline_bin] += 1

    # Normalize by column (what % of each baseline score goes where)
    transition_matrix_norm = transition_matrix.copy()
    for col in range(10):
        col_sum = transition_matrix[:, col].sum()
        if col_sum > 0:
            transition_matrix_norm[:, col] = transition_matrix[:, col] / col_sum * 100

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 18))

    # Increase annotation font size for all heatmaps
    annot_kws = {"fontsize": 12, "fontweight": "bold"}

    # Plot 1: Raw counts
    sns.heatmap(transition_matrix, annot=True, fmt='.0f', cmap='YlOrRd',
                ax=axes[0, 0], cbar_kws={'label': 'Count (log scale)'},
                norm=LogNorm(vmin=0.1, vmax=max(transition_matrix.max(), 1)),
                xticklabels=bin_labels, yticklabels=bin_labels,
                linewidths=0.5, linecolor='gray', annot_kws=annot_kws)

    axes[0, 0].set_xlabel('Baseline Harmfulness', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel(f'{transition_name} Harmfulness', fontsize=14, fontweight='bold')
    axes[0, 0].set_title(f'Baseline → {transition_name} Transitions (Counts)', fontsize=16, fontweight='bold')
    axes[0, 0].tick_params(axis='both', which='major', labelsize=12)
    axes[0, 0].invert_yaxis()

    # Add diagonal (no change)
    for i in range(10):
        axes[0, 0].add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                                           edgecolor='blue', lw=2, alpha=0.5))

    # Add risk category lines
    axes[0, 0].axvline(x=4, color='green', linestyle='--', alpha=0.3, linewidth=1.5)
    axes[0, 0].axhline(y=6, color='green', linestyle='--', alpha=0.3, linewidth=1.5)
    axes[0, 0].axvline(x=7, color='orange', linestyle='--', alpha=0.3, linewidth=1.5)
    axes[0, 0].axhline(y=3, color='orange', linestyle='--', alpha=0.3, linewidth=1.5)

    # Plot 2: Percentage normalized
    sns.heatmap(transition_matrix_norm, annot=transition_matrix, fmt='.0f', cmap='YlGnBu',
                ax=axes[0, 1], cbar_kws={'label': 'Percentage (%)'},
                xticklabels=bin_labels, yticklabels=bin_labels,
                linewidths=0.5, linecolor='gray', annot_kws=annot_kws)
    axes[0, 1].set_xlabel('Baseline Harmfulness', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel(f'{transition_name} Harmfulness', fontsize=14, fontweight='bold')
    axes[0, 1].set_title(f'Baseline → {transition_name} (% of Baseline)', fontsize=16, fontweight='bold')
    axes[0, 1].tick_params(axis='both', which='major', labelsize=12)
    axes[0, 1].invert_yaxis()

    # Add diagonal
    for i in range(10):
        axes[0, 1].add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                                           edgecolor='blue', lw=2, alpha=0.5))

    # Plot 3: Directional movement (commitment bias visualization)
    net_movement = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            if i < j:  # Below diagonal = decrease (commitment bias)
                net_movement[i, j] = transition_matrix_norm[i, j]
            elif i > j:  # Above diagonal = increase (correction)
                net_movement[i, j] = -transition_matrix_norm[i, j]

    sns.heatmap(net_movement, annot=True, fmt='.0f', cmap='RdYlGn_r',
                ax=axes[1, 0], cbar_kws={'label': 'Direction (Red=Decrease, Green=Increase)'},
                xticklabels=bin_labels, yticklabels=bin_labels,
                linewidths=0.5, linecolor='gray', center=0, vmin=-50, vmax=50, annot_kws=annot_kws)
    axes[1, 0].set_xlabel('Baseline Harmfulness Score', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel(f'{transition_name} Harmfulness Score', fontsize=14, fontweight='bold')
    axes[1, 0].set_title('Commitment Bias Pattern\n(Red = Downplaying risk)', fontsize=16, fontweight='bold')
    axes[1, 0].tick_params(axis='both', which='major', labelsize=12)
    axes[1, 0].invert_yaxis()

    # Add diagonal
    for i in range(10):
        axes[1, 0].add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                                           edgecolor='black', lw=2, alpha=0.7))

    # Plot 4: Simplified risk category transitions
    risk_categories = {
        'Low (0-40)': (0, 4),
        'Medium (40-70)': (4, 7),
        'High (70-100)': (7, 10)
    }

    category_matrix = np.zeros((3, 3))
    for baseline, target in all_transitions_list:
        baseline_cat = 0 if baseline <= 40 else (1 if baseline <= 70 else 2)
        target_cat = 0 if target <= 40 else (1 if target <= 70 else 2)
        category_matrix[target_cat, baseline_cat] += 1

    # Normalize
    category_matrix_norm = category_matrix.copy()
    for col in range(3):
        col_sum = category_matrix[:, col].sum()
        if col_sum > 0:
            category_matrix_norm[:, col] = category_matrix[:, col] / col_sum * 100

    sns.heatmap(category_matrix_norm, annot=True, fmt='.1f', cmap='coolwarm',
                ax=axes[1, 1], cbar_kws={'label': 'Percentage (%)'},
                xticklabels=list(risk_categories.keys()),
                yticklabels=list(risk_categories.keys()),
                linewidths=2, linecolor='black', vmin=0, vmax=100,
                annot_kws={"fontsize": 14, "fontweight": "bold"})
    axes[1, 1].set_xlabel('Baseline Risk Category', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel(f'{transition_name} Risk Category', fontsize=14, fontweight='bold')
    axes[1, 1].set_title('Risk Category Transitions\n(Simplified view)', fontsize=16, fontweight='bold')
    axes[1, 1].tick_params(axis='both', which='major', labelsize=12)
    axes[1, 1].invert_yaxis()

    # Highlight diagonal (no category change)
    for i in range(3):
        axes[1, 1].add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                                           edgecolor='green', lw=3, alpha=0.7))

    # Increase colorbar label font size
    for ax in axes.flat:
        cbar = ax.collections[0].colorbar
        if cbar:
            cbar.ax.tick_params(labelsize=11)
            cbar.set_label(cbar.ax.get_ylabel(), fontsize=12)

    plt.suptitle(f'Harmfulness Score Transition Analysis - All Models Combined\n(Baseline → {transition_name})',
                 fontsize=18, fontweight='bold', y=0.99)
    plt.tight_layout()

    filename = f'harmfulness_transition_heatmap_{transition_name.lower()}_combined.png'
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

    print(f"  ✓ Saved {filename}")


def create_individual_model_heatmaps(all_transitions, transition_type, transition_name, bin_labels, output_dir):
    """Create individual model heatmaps."""

    # Sort models alphabetically
    models = sorted(all_transitions.keys())

    # Create figure with subplots for top 6 models
    top_models = models[:6] if len(models) >= 6 else models

    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    axes = axes.flatten()

    annot_kws = {"fontsize": 11, "fontweight": "bold"}

    for idx, model_name in enumerate(top_models):
        if idx >= 6:
            break

        model_data = all_transitions[model_name]
        transitions = model_data[transition_type]

        # Create transition matrix
        transition_matrix = np.zeros((10, 10))
        for baseline, target in transitions:
            baseline_bin = min(int(baseline / 10), 9)
            target_bin = min(int(target / 10), 9)
            transition_matrix[target_bin, baseline_bin] += 1

        # Normalize
        transition_matrix_norm = transition_matrix.copy()
        for col in range(10):
            col_sum = transition_matrix[:, col].sum()
            if col_sum > 0:
                transition_matrix_norm[:, col] = transition_matrix[:, col] / col_sum * 100

        ax = axes[idx]
        sns.heatmap(transition_matrix_norm, annot=transition_matrix, fmt='.0f', cmap='YlGnBu',
                    ax=ax, cbar_kws={'label': '%'},
                    xticklabels=bin_labels, yticklabels=bin_labels,
                    linewidths=0.5, linecolor='gray', annot_kws=annot_kws)
        ax.set_xlabel('Baseline', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{transition_name}', fontsize=12, fontweight='bold')
        ax.set_title(f'{model_name}\n(n={len(transitions)})', fontsize=13, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.invert_yaxis()

        # Add diagonal
        for i in range(10):
            ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                                       edgecolor='blue', lw=1.5, alpha=0.5))

    # Increase colorbar label font size for all subplots
    for ax in axes[:len(top_models)]:
        cbar = ax.collections[0].colorbar
        if cbar:
            cbar.ax.tick_params(labelsize=10)
            cbar.set_label(cbar.ax.get_ylabel(), fontsize=11)

    plt.suptitle(f'Model-Specific Harmfulness Transition Patterns\n(Baseline → {transition_name})',
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()

    filename = f'harmfulness_transition_heatmap_{transition_name.lower()}_by_model.png'
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

    print(f"  ✓ Saved {filename}")


def create_comparison_heatmap(all_transitions, bin_labels, output_dir):
    """Create a comparison heatmap showing differences between followup and continuation patterns."""

    # Combine all transitions
    all_followup = []
    all_continuation = []

    for model_data in all_transitions.values():
        all_followup.extend(model_data['baseline_to_followup'])
        all_continuation.extend(model_data['baseline_to_continuation'])

    if not all_followup or not all_continuation:
        print("  Insufficient data for comparison heatmap")
        return

    # Create transition matrices
    followup_matrix = np.zeros((10, 10))
    continuation_matrix = np.zeros((10, 10))

    for baseline, followup in all_followup:
        baseline_bin = min(int(baseline / 10), 9)
        followup_bin = min(int(followup / 10), 9)
        followup_matrix[followup_bin, baseline_bin] += 1

    for baseline, continuation in all_continuation:
        baseline_bin = min(int(baseline / 10), 9)
        continuation_bin = min(int(continuation / 10), 9)
        continuation_matrix[continuation_bin, baseline_bin] += 1

    # Normalize both matrices
    followup_norm = followup_matrix.copy()
    continuation_norm = continuation_matrix.copy()

    for col in range(10):
        if followup_matrix[:, col].sum() > 0:
            followup_norm[:, col] = followup_matrix[:, col] / followup_matrix[:, col].sum() * 100
        if continuation_matrix[:, col].sum() > 0:
            continuation_norm[:, col] = continuation_matrix[:, col] / continuation_matrix[:, col].sum() * 100

    # Calculate difference (continuation - followup)
    difference_matrix = continuation_norm - followup_norm

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    annot_kws = {"fontsize": 12, "fontweight": "bold"}

    # Plot 1: Followup normalized
    sns.heatmap(followup_norm, annot=True, fmt='.0f', cmap='Blues',
                ax=axes[1], cbar_kws={'label': 'Counts'},
                xticklabels=bin_labels, yticklabels=bin_labels,
                linewidths=0.5, linecolor='gray', annot_kws=annot_kws)
    axes[1].set_xlabel('Baseline', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Same Turn Self Attributed', fontsize=14, fontweight='bold')
    axes[1].set_title('Baseline → Same Turn Self Attributed', fontsize=16, fontweight='bold')
    axes[1].tick_params(axis='both', which='major', labelsize=11)
    axes[1].invert_yaxis()

    # Plot 2: Continuation normalized
    sns.heatmap(continuation_norm, annot=continuation_matrix, fmt='.0f', cmap='Greens',
                ax=axes[0], cbar_kws={'label': 'Counts'},
                xticklabels=bin_labels, yticklabels=bin_labels,
                linewidths=0.5, linecolor='gray', annot_kws=annot_kws)
    axes[0].set_xlabel('Baseline', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Previous Turn Self Attributed', fontsize=14, fontweight='bold')
    axes[0].set_title('Baseline → Previous Turn Self Attributed', fontsize=16, fontweight='bold')
    axes[0].tick_params(axis='both', which='major', labelsize=11)
    axes[0].invert_yaxis()

    # Plot 3: Difference (Continuation - Followup)
    sns.heatmap(difference_matrix, annot=followup_matrix, fmt='.0f', cmap='RdBu_r',
                ax=axes[2], cbar_kws={'label': 'Change'},
                xticklabels=bin_labels, yticklabels=bin_labels,
                linewidths=0.5, linecolor='gray', center=0, vmin=-30, vmax=30, annot_kws=annot_kws)
    axes[2].set_xlabel('Baseline', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Target', fontsize=14, fontweight='bold')
    axes[2].set_title('Difference\n(Same Turn - Previous Turn)', fontsize=16, fontweight='bold')
    axes[2].tick_params(axis='both', which='major', labelsize=11)
    axes[2].invert_yaxis()

    # Add diagonals
    for ax in axes:
        for i in range(10):
            ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                                       edgecolor='black', lw=1.5, alpha=0.5))

    # Increase colorbar label font size
    for ax in axes:
        cbar = ax.collections[0].colorbar
        if cbar:
            cbar.ax.tick_params(labelsize=11)
            cbar.set_label(cbar.ax.get_ylabel(), fontsize=12)

    plt.suptitle('Same Turn vs Previous Turn Transitions',
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()

    filename = 'harmfulness_transition_comparison.png'
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

    print(f"  ✓ Saved {filename}")


def print_transition_statistics(all_transitions):
    """Print detailed statistics for both transition types."""

    print("\n" + "=" * 60)
    print("HARMFULNESS TRANSITION ANALYSIS SUMMARY")
    print("=" * 60)

    # Analyze FOLLOWUP transitions
    all_followup = []
    for model_data in all_transitions.values():
        all_followup.extend(model_data['baseline_to_followup'])

    if all_followup:
        print("\n--- BASELINE → FOLLOWUP TRANSITIONS ---")
        analyze_transitions(all_followup, "Same Turn Self Attributed")

    # Analyze CONTINUATION transitions
    all_continuation = []
    for model_data in all_transitions.values():
        all_continuation.extend(model_data['baseline_to_continuation'])

    if all_continuation:
        print("\n--- BASELINE → CONTINUATION TRANSITIONS ---")
        analyze_transitions(all_continuation, "Previous Turn Self Attributed")

    # Compare the two if both exist
    if all_followup and all_continuation:
        print("\n--- COMPARISON: FOLLOWUP vs CONTINUATION ---")

        followup_shifts = np.array([f - b for b, f in all_followup])
        continuation_shifts = np.array([c - b for b, c in all_continuation])

        print(
            f"Mean shift difference (Continuation - Followup): {np.mean(continuation_shifts) - np.mean(followup_shifts):.2f}")
        print(f"Followup mean shift: {np.mean(followup_shifts):.2f}")
        print(f"Continuation mean shift: {np.mean(continuation_shifts):.2f}")

        # Compare high-risk scenarios
        high_risk_followup = [(b, f) for b, f in all_followup if b > 70]
        high_risk_continuation = [(b, c) for b, c in all_continuation if b > 70]

        if high_risk_followup and high_risk_continuation:
            hr_followup_shifts = [f - b for b, f in high_risk_followup]
            hr_continuation_shifts = [c - b for b, c in high_risk_continuation]

            print(f"\nHigh-risk scenarios (baseline > 70):")
            print(f"  Followup mean shift: {np.mean(hr_followup_shifts):.2f}")
            print(f"  Continuation mean shift: {np.mean(hr_continuation_shifts):.2f}")
            print(f"  Difference: {np.mean(hr_continuation_shifts) - np.mean(hr_followup_shifts):.2f}")

    # Per-model statistics
    print("\n" + "-" * 40)
    print("PER-MODEL STATISTICS")
    print("-" * 40)

    for model_name in sorted(all_transitions.keys()):
        model_data = all_transitions[model_name]
        followup_transitions = model_data['baseline_to_followup']
        continuation_transitions = model_data['baseline_to_continuation']

        print(f"\n{model_name}:")

        if followup_transitions:
            shifts = [f - b for b, f in followup_transitions]
            print(f"  Followup:")
            print(f"    Samples: {len(followup_transitions)}")
            print(f"    Mean shift: {np.mean(shifts):.2f}")
            print(f"    Median shift: {np.median(shifts):.2f}")

            # Check for concerning patterns
            high_risk = [(b, f) for b, f in followup_transitions if b > 70]
            if high_risk:
                high_risk_shifts = [f - b for b, f in high_risk]
                high_risk_mean = np.mean(high_risk_shifts)
                print(f"    High-risk mean shift: {high_risk_mean:.2f}")
                if high_risk_mean < -15:
                    print(f"    ⚠️  Significant commitment bias in high-risk scenarios!")

        if continuation_transitions:
            shifts = [c - b for b, c in continuation_transitions]
            print(f"  Continuation:")
            print(f"    Samples: {len(continuation_transitions)}")
            print(f"    Mean shift: {np.mean(shifts):.2f}")
            print(f"    Median shift: {np.median(shifts):.2f}")

            # Check for concerning patterns
            high_risk = [(b, c) for b, c in continuation_transitions if b > 70]
            if high_risk:
                high_risk_shifts = [c - b for b, c in high_risk]
                high_risk_mean = np.mean(high_risk_shifts)
                print(f"    High-risk mean shift: {high_risk_mean:.2f}")
                if high_risk_mean < -15:
                    print(f"    ⚠️  Significant commitment bias in high-risk scenarios!")


def analyze_transitions(transitions, transition_name):
    """Analyze and print statistics for a set of transitions."""

    shifts = np.array([target - baseline for baseline, target in transitions])

    print(f"\nTotal transitions analyzed: {len(shifts)}")
    print(f"Mean change: {np.mean(shifts):.2f}")
    print(f"Median change: {np.median(shifts):.2f}")
    print(f"Std deviation: {np.std(shifts):.2f}")

    # Categorize changes
    decreases_large = np.sum(shifts < -20)
    decreases_small = np.sum((shifts < -5) & (shifts >= -20))
    stable = np.sum(np.abs(shifts) <= 5)
    increases_small = np.sum((shifts > 5) & (shifts <= 20))
    increases_large = np.sum(shifts > 20)

    print(f"\nTransition patterns:")
    print(f"  Large decrease (< -20): {decreases_large} ({100 * decreases_large / len(shifts):.1f}%)")
    print(f"  Small decrease (-20 to -5): {decreases_small} ({100 * decreases_small / len(shifts):.1f}%)")
    print(f"  Stable (±5): {stable} ({100 * stable / len(shifts):.1f}%)")
    print(f"  Small increase (5 to 20): {increases_small} ({100 * increases_small / len(shifts):.1f}%)")
    print(f"  Large increase (> 20): {increases_large} ({100 * increases_large / len(shifts):.1f}%)")

    # Analyze high-risk downplaying
    high_risk_transitions = [(b, t) for b, t in transitions if b > 70]
    if high_risk_transitions:
        high_risk_shifts = [t - b for b, t in high_risk_transitions]
        downplaying_count = sum(1 for s in high_risk_shifts if s < -10)
        print(f"\nHigh-risk scenarios (baseline > 70):")
        print(f"  Total: {len(high_risk_transitions)}")
        print(f"  Mean shift: {np.mean(high_risk_shifts):.2f}")
        print(
            f"  Downplayed (shift < -10): {downplaying_count} ({100 * downplaying_count / len(high_risk_transitions):.1f}%)")
        if np.mean(high_risk_shifts) < -10:
            print(f"  ⚠️  WARNING: Systematic downplaying of high-risk scenarios detected in {transition_name}!")


def main():
    """Main execution"""
    import sys

    # Get log directory from command line or use default
    log_directory = sys.argv[1] if len(sys.argv) > 1 else "./log"

    print(f"Creating harmfulness transition heatmaps from: {log_directory}")

    # Create transition heatmaps
    create_transition_heatmaps(log_directory, output_dir="plots")

    print("\n✓ Harmfulness transition heatmap visualizations complete!")
    print("Check the 'plots' directory for generated heatmaps.")


if __name__ == "__main__":
    main()