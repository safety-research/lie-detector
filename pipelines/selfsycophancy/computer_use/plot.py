import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from inspect_ai.log import read_eval_log, EvalLog
from typing import Dict, List, Optional, Tuple


def extract_risk_scores_by_category(log_path: str) -> Dict:
    """
    Extract risk assessment scores from an Inspect AI evaluation log,
    categorized by true risk level.

    Parameters:
    -----------
    log_path : str
        Path to the Inspect AI log file

    Returns:
    --------
    dict : Dictionary containing risk scores categorized by true risk level
    """
    # Read the evaluation log
    log = read_eval_log(log_path)

    # Extract model name from log
    model_name = log.eval.model

    # Initialize collectors for different risk categories
    categories = {
        'low': {'baseline': [], 'prefilled': [], 'continuation': [], 'true_risks': []},
        'medium': {'baseline': [], 'prefilled': [], 'continuation': [], 'true_risks': []},
        'high': {'baseline': [], 'prefilled': [], 'continuation': [], 'true_risks': []}
    }

    # Get samples
    samples = log.samples

    # Iterate through samples to extract risk scores
    for idx, sample in enumerate(samples):
        metadata = sample.metadata

        # Get true risk level from target or metadata
        true_risk = None
        if hasattr(sample, 'target') and sample.target:
            try:
                true_risk = float(sample.target)
            except:
                pass
        if true_risk is None and 'true_risk_level' in metadata:
            true_risk = float(metadata['true_risk_level'])

        if true_risk is None:
            continue  # Skip if we can't determine true risk

        # Categorize by true risk level
        if true_risk <= 40:
            category = 'low'
        elif true_risk <= 70:
            category = 'medium'
        else:
            category = 'high'

        categories[category]['true_risks'].append(true_risk)

        # Extract baseline scores
        if 'baseline_mean' in metadata:
            categories[category]['baseline'].append(metadata['baseline_mean'])
        elif 'baseline_risks' in metadata:
            categories[category]['baseline'].append(np.mean(metadata['baseline_risks']))

        # Extract prefilled scores
        if 'prefilled_mean' in metadata:
            categories[category]['prefilled'].append(metadata['prefilled_mean'])
        elif 'prefilled_risk_score' in metadata:
            categories[category]['prefilled'].append(metadata['prefilled_risk_score'])
        elif 'prefilled_risks' in metadata:
            categories[category]['prefilled'].append(np.mean(metadata['prefilled_risks']))

        # Extract continuation scores
        if 'continuation_risk_score' in metadata:
            categories[category]['continuation'].append(metadata['continuation_risk_score'])

    # Calculate statistics for each category
    results = {
        'model_name': model_name,
        'categories': {}
    }

    for cat_name, cat_data in categories.items():
        if len(cat_data['baseline']) > 0:  # Only include categories with data
            results['categories'][cat_name] = {
                'n_samples': len(cat_data['baseline']),
                'true_risk_mean': np.mean(cat_data['true_risks']) if cat_data['true_risks'] else 0,
                'baseline_mean': np.mean(cat_data['baseline']) if cat_data['baseline'] else 0,
                #'baseline_std': np.std(cat_data['baseline'], ddof=1) if cat_data['baseline'] else 0,
                'baseline_std': np.std(cat_data['baseline'], ddof=1) / np.sqrt(len(cat_data['baseline'])) if cat_data[
                    'baseline'] else 0,

                'prefilled_mean': np.mean(cat_data['prefilled']) if cat_data['prefilled'] else 0,
                #'prefilled_std': np.std(cat_data['prefilled']) if cat_data['prefilled'] else 0,
                'prefilled_std': np.std(cat_data['prefilled'], ddof=1) / np.sqrt(len(cat_data['prefilled'])) if cat_data[
                    'prefilled'] else 0,

                'continuation_mean': np.mean(cat_data['continuation']) if cat_data['continuation'] else 0,
               # 'continuation_std': np.std(cat_data['continuation']) if cat_data['continuation'] else 0,

                'continuation_std': np.std(cat_data['continuation'], ddof=1) / np.sqrt(len(cat_data['continuation'])) if cat_data[
                    'continuation'] else 0,
            }

    return results


def load_all_logs_by_category(log_directory: str) -> Dict[str, Dict]:
    """
    Load all evaluation logs from a directory, categorized by risk level.
    """
    log_dir = Path(log_directory)
    all_results = {}

    # Find all log files
    log_files = list(log_dir.glob("*.eval"))
    if not log_files:
        log_files = list(log_dir.glob("*.json"))

    print(f"Found {len(log_files)} log files in {log_directory}")

    for log_file in log_files:
        print(f"Processing: {log_file.name}")
        try:
            results = extract_risk_scores_by_category(str(log_file))
            model_name = results['model_name']

            # Clean up model name if needed
            if '/' in model_name:
                model_name = model_name.split('/')[-1]

            all_results[model_name] = results
            total_samples = sum(cat['n_samples'] for cat in results['categories'].values())
            print(f"  ✓ Extracted data for {model_name}: {total_samples} total samples")
            for cat_name, cat_data in results['categories'].items():
                print(f"    - {cat_name}: {cat_data['n_samples']} samples")

        except Exception as e:
            print(f"  ✗ Error processing {log_file.name}: {e}")

    return all_results


def create_categorized_commitment_bias_charts(results: Dict, save_dir: str = './'):
    """
    Create separate bar charts for each risk category and model.
    """
    if not results:
        print("No data to plot!")
        return

    # Risk category labels and colors
    risk_categories = {
        'low': {'label': 'Low Risk (0-40)', 'color': '#28a745'},
        'medium': {'label': 'Medium Risk (41-70)', 'color': '#ffc107'},
        'high': {'label': 'High Risk (71-100)', 'color': '#dc3545'}
    }

    # Colors for conditions
    condition_colors = {
        'baseline': '#2E86AB',  # Blue
        'prefilled': '#A23B72',  # Purple
        'continuation': '#F18F01'  # Orange
    }

    # Create a figure for each model
    for model_name, model_data in results.items():
        categories = model_data['categories']

        # Count how many categories have data
        valid_categories = [cat for cat in ['low', 'medium', 'high'] if cat in categories]
        n_categories = len(valid_categories)

        if n_categories == 0:
            print(f"No valid categories for {model_name}")
            continue

        # Create subplots for this model
        fig, axes = plt.subplots(1, n_categories, figsize=(5 * n_categories, 6), sharey=True)

        # Ensure axes is iterable
        if n_categories == 1:
            axes = [axes]

        # Plot each risk category
        for idx, (cat_name, ax) in enumerate(zip(valid_categories, axes)):
            cat_data = categories[cat_name]

            # Prepare data
            conditions = []
            means = []
            stds = []
            colors = []

            # Always show baseline if available
            if cat_data['baseline_mean'] > 0:
                conditions.append('Baseline')
                means.append(cat_data['baseline_mean'])
                stds.append(cat_data['baseline_std'])
                colors.append(condition_colors['baseline'])

            # Show prefilled if available
            if cat_data['prefilled_mean'] > 0:
                conditions.append('Followup')
                means.append(cat_data['prefilled_mean'])
                stds.append(cat_data['prefilled_std'])
                colors.append(condition_colors['prefilled'])

            # Show continuation if available
            if cat_data['continuation_mean'] > 0:
                conditions.append('Continuation')
                means.append(cat_data['continuation_mean'])
                stds.append(cat_data['continuation_std'])
                colors.append(condition_colors['continuation'])

            # Create bars
            x_pos = np.arange(len(conditions))
            bars = ax.bar(x_pos, means, color=colors, alpha=0.8,
                          edgecolor='black', linewidth=1.5)

            # Add error bars
            ax.errorbar(x_pos, means, yerr=stds, fmt='none',
                        ecolor='black', capsize=5, capthick=2)

            # Add value labels on bars
            # for bar, mean, std in zip(bars, means, stds):
            #     height = bar.get_height()
            #     ax.text(bar.get_x() + bar.get_width() / 2., height + std + 2,
            #             f'{mean:.1f}',
            #             ha='center', va='bottom', fontsize=10, fontweight='bold')

            # Customize subplot
            ax.set_title(risk_categories[cat_name]['label'],
                         fontsize=13, fontweight='bold',
                         color=risk_categories[cat_name]['color'])
            ax.set_xlabel('Assessment Condition', fontsize=11)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(conditions)
            ax.set_ylim(0, 100)
            ax.grid(axis='y', alpha=0.3, linestyle='--')

            # Add reference line for true risk mean
            true_risk_mean = cat_data['true_risk_mean']
            ax.axhline(y=true_risk_mean, color='red', linestyle=':', alpha=0.7,
                       label=f'True Risk Mean: {true_risk_mean:.0f}')

            # Only show y-label on leftmost plot
            if idx == 0:
                ax.set_ylabel('Harmfulness Rating', fontsize=11)

            # REMOVED: Sample size text box
            # n_samples = cat_data['n_samples']
            # ax.text(0.02, 0.98, f'n = {n_samples}',
            #         transform=ax.transAxes, fontsize=9,
            #         verticalalignment='top',
            #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Calculate and display commitment bias
            if cat_data['baseline_mean'] > 0 and cat_data['prefilled_mean'] > 0:
                bias = cat_data['baseline_mean'] - cat_data['prefilled_mean']
                bias_text = f'Bias: {bias:+.1f}'

                # Color based on bias magnitude and risk level
                if cat_name == 'high' and bias > 10:
                    bias_color = 'red'  # Concerning - downplaying high risk
                elif bias > 15:
                    bias_color = 'orange'
                else:
                    bias_color = 'black'

                ax.text(0.5, 0.02, bias_text,
                        transform=ax.transAxes, fontsize=10,
                        color=bias_color, fontweight='bold' if abs(bias) > 10 else 'normal',
                        horizontalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

            # Add legend for this subplot - positioned below
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
                      fontsize=12, ncol=1)

        # Overall figure title
        fig.suptitle(f'{model_name} - Risk Assessment by Category',
                     fontsize=15, fontweight='bold', y=1.02)

        plt.tight_layout()

        # Save the figure
        save_path = Path(save_dir) / f'{model_name.replace("/", "_")}_categorized_risk.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
        plt.show()

        print(f"Saved chart for {model_name} to {save_path}")


def create_high_risk_comparison_chart(results: Dict, save_path: str = 'high_risk_comparison.png'):
    """
    Create a focused comparison chart showing only high-risk situations across all models.
    """
    if not results:
        print("No data to plot!")
        return

    # Extract high-risk data for all models
    models_with_high_risk = []
    high_risk_data = {}

    for model_name, model_data in results.items():
        if 'high' in model_data['categories']:
            models_with_high_risk.append(model_name)
            high_risk_data[model_name] = model_data['categories']['high']

    if not models_with_high_risk:
        print("No high-risk data found in any model!")
        return

    # Sort models alphabetically
    models_with_high_risk.sort()

    # Setup the figure
    fig, ax = plt.subplots(figsize=(max(12, len(models_with_high_risk) * 2), 8))

    # Bar settings
    bar_width = 0.25
    x = np.arange(len(models_with_high_risk))

    # Colors for conditions
    colors = {
        'baseline': '#2E86AB',
        'followup': '#F18F01',
        'continuation': '#A23B72'
    }

    # Collect data for each condition
    baseline_means = []
    baseline_stds = []
    followup_means = []
    followup_stds = []
    continuation_means = []
    continuation_stds = []
    true_risk_means = []
    sample_sizes = []

    for model in models_with_high_risk:
        data = high_risk_data[model]
        baseline_means.append(data['baseline_mean'])
        baseline_stds.append(data['baseline_std'])
        followup_means.append(data['prefilled_mean'] if data['prefilled_mean'] > 0 else np.nan)
        followup_stds.append(data['prefilled_std'] if data['prefilled_mean'] > 0 else 0)
        continuation_means.append(data['continuation_mean'] if data['continuation_mean'] > 0 else np.nan)
        continuation_stds.append(data['continuation_std'] if data['continuation_mean'] > 0 else 0)
        true_risk_means.append(data['true_risk_mean'])
        sample_sizes.append(data['n_samples'])

    # Create bars
    bars1 = ax.bar(x - bar_width, baseline_means, bar_width,
                   yerr=baseline_stds, label='Baseline',
                   color=colors['baseline'], alpha=0.8,
                   edgecolor='black', linewidth=1.5,
                   capsize=4, error_kw={'linewidth': 1.5})

    # Only plot followup bars where data exists
    followup_x = [i - bar_width for i, val in enumerate(followup_means) if not np.isnan(val)]
    followup_y = [val for val in followup_means if not np.isnan(val)]
    followup_err = [followup_stds[i] for i, val in enumerate(followup_means) if not np.isnan(val)]
    if followup_y:
        bars2 = ax.bar(x, followup_y, bar_width,
                       yerr=followup_err, label='Followup',
                       color=colors['followup'], alpha=0.8,
                       edgecolor='black', linewidth=1.5,
                       capsize=4, error_kw={'linewidth': 1.5})

    # Only plot continuation bars where data exists
    continuation_x = [i + bar_width for i, val in enumerate(continuation_means) if not np.isnan(val)]
    continuation_y = [val for val in continuation_means if not np.isnan(val)]
    continuation_err = [continuation_stds[i] for i, val in enumerate(continuation_means) if not np.isnan(val)]
    if continuation_y:
        bars3 = ax.bar(continuation_x, continuation_y, bar_width,
                       yerr=continuation_err, label='Continuation',
                       color=colors['continuation'], alpha=0.8,
                       edgecolor='black', linewidth=1.5,
                       capsize=4, error_kw={'linewidth': 1.5})

    # Add value labels on bars
    # def add_value_labels(bars, values, stds):
    #     for bar, val, std in zip(bars, values, stds):
    #         height = bar.get_height()
    #         ax.text(bar.get_x() + bar.get_width() / 2., height + std + 1,
    #                 f'{val:.1f}', ha='center', va='bottom',
    #                 fontsize=9, fontweight='bold')
    #
    # add_value_labels(bars1, baseline_means, baseline_stds)
    # if followup_y:
    #     # Need to get the correct bars for followup
    #     bars2_list = [b for b in bars2]
    #     add_value_labels(bars2_list, followup_y, followup_err)

    # REMOVED: Sample sizes below each model
    # for i, model in enumerate(models_with_high_risk):
    #     ax.text(i, -5, f'n={sample_sizes[i]}',
    #             ha='center', va='top', fontsize=8, color='gray')

    # Add true risk mean line
    ax.axhline(y=np.mean(true_risk_means), color='red', linestyle='--',
               alpha=0.7, linewidth=2, label=f'Avg True Risk: {np.mean(true_risk_means):.1f}')

    # Customize the plot
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Harmfulness Rating', fontsize=12, fontweight='bold')
    ax.set_title('Models Systematically Underestimate Harmfulness via Commitment Bias\nIn High-Risk Computer Use Settings',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    # CHANGED: Removed rotation to make labels horizontal
    ax.set_xticklabels(models_with_high_risk)
    # Move legend to bottom center, outside the plot area
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              fontsize=12, framealpha=0.9, ncol=4, columnspacing=2)
    ax.set_ylim(0, 105)  # Changed from -8 to 0 since we removed the n= labels below
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add a text box with key insights
    # textstr = 'Red bias values indicate concerning\ndownplaying of high-risk actions'
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    # ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
    #         verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.show()

    print(f"Saved high-risk comparison chart to {save_path}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("HIGH-RISK SITUATION SUMMARY")
    print("=" * 60)
    for model in models_with_high_risk:
        data = high_risk_data[model]
        print(f"\n{model}:")
        print(f"  Baseline: {data['baseline_mean']:.1f} ± {data['baseline_std']:.1f}")
        if data['prefilled_mean'] > 0:
            bias = data['baseline_mean'] - data['prefilled_mean']
            print(f"  Followup: {data['prefilled_mean']:.1f} ± {data['prefilled_std']:.1f}")
            print(f"  Commitment Bias: {bias:+.1f}")
            if bias > 15:
                print(f"  ⚠️  WARNING: Significant downplaying of high-risk actions!")


def create_comparison_matrix(results: Dict, save_path: str = 'risk_matrix_comparison.png'):
    """
    Create a matrix visualization comparing all models across risk categories.
    """
    if not results:
        return

    # Sort models alphabetically
    models = sorted(list(results.keys()))
    categories = ['low', 'medium', 'high']

    # Create figure with subplots
    fig, axes = plt.subplots(3, len(models), figsize=(4 * len(models), 10), sharey='row')

    # Ensure axes is 2D
    if len(models) == 1:
        axes = axes.reshape(-1, 1)

    # Category info
    cat_info = {
        'low': {'label': 'Low Risk\n(0-40)', 'color': '#28a745'},
        'medium': {'label': 'Medium Risk\n(41-70)', 'color': '#ffc107'},
        'high': {'label': 'High Risk\n(71-100)', 'color': '#dc3545'}
    }

    # Plot each model and category
    for col_idx, model_name in enumerate(models):
        model_data = results[model_name]

        for row_idx, cat_name in enumerate(categories):
            ax = axes[row_idx, col_idx]

            if cat_name not in model_data['categories']:
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                        ha='center', va='center', fontsize=12, color='gray')
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                cat_data = model_data['categories'][cat_name]

                # Prepare data
                conditions = ['Baseline', 'Followup', 'Continuation']
                means = [
                    cat_data.get('baseline_mean', 0),
                    cat_data.get('prefilled_mean', 0),
                    cat_data.get('continuation_mean', 0)
                ]
                stds = [
                    cat_data.get('baseline_std', 0),
                    cat_data.get('prefilled_std', 0),
                    cat_data.get('continuation_std', 0)
                ]

                # Filter out zero values
                valid_data = [(c, m, s) for c, m, s in zip(conditions, means, stds) if m > 0]
                if valid_data:
                    conditions, means, stds = zip(*valid_data)

                    x_pos = np.arange(len(conditions))
                    colors = ['#2E86AB', '#A23B72', '#F18F01'][:len(conditions)]

                    bars = ax.bar(x_pos, means, color=colors, alpha=0.7,
                                  edgecolor='black', linewidth=1)
                    ax.errorbar(x_pos, means, yerr=stds, fmt='none',
                                ecolor='black', capsize=3, capthick=1)

                    # Value labels
                    for bar, mean in zip(bars, means):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                                f'{mean:.0f}', ha='center', va='bottom', fontsize=8)

                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(conditions, rotation=45, ha='right', fontsize=8)
                    ax.set_ylim(0, 100)
                    ax.grid(axis='y', alpha=0.2)

                    # REMOVED: Sample size text
                    # n = cat_data.get('n_samples', 0)
                    # ax.text(0.02, 0.95, f'n={n}', transform=ax.transAxes,
                    #         fontsize=7, va='top')

            # Add labels
            if col_idx == 0:
                ax.set_ylabel(cat_info[cat_name]['label'], fontsize=11,
                              color=cat_info[cat_name]['color'], fontweight='bold')

            if row_idx == 0:
                ax.set_title(model_name, fontsize=11, fontweight='bold')

    fig.suptitle('Risk Assessment Commitment Bias - All Models by Risk Category',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.show()
    print(f"Saved comparison matrix to {save_path}")


def print_categorized_summary(results: Dict):
    """
    Print summary statistics by category.
    """
    print("\n" + "=" * 70)
    print("CATEGORIZED COMMITMENT BIAS SUMMARY")
    print("=" * 70 + "\n")

    # Sort models alphabetically for consistent output
    sorted_models = sorted(results.keys())

    for model_name in sorted_models:
        model_data = results[model_name]
        print(f"\n{model_name}")
        print("-" * len(model_name))

        for cat_name in ['low', 'medium', 'high']:
            if cat_name not in model_data['categories']:
                continue

            cat_data = model_data['categories'][cat_name]

            print(f"\n  {cat_name.upper()} RISK (True mean: {cat_data['true_risk_mean']:.1f}):")
            print(f"    Samples: {cat_data['n_samples']}")
            print(f"    Baseline: {cat_data['baseline_mean']:.1f} ± {cat_data['baseline_std']:.1f}")

            if cat_data['prefilled_mean'] > 0:
                bias = cat_data['baseline_mean'] - cat_data['prefilled_mean']
                print(f"    Followup: {cat_data['prefilled_mean']:.1f} ± {cat_data['prefilled_std']:.1f}")
                print(f"    → Bias: {bias:+.1f}")

                # Interpretation
                if cat_name == 'high' and bias > 15:
                    print(f"    ⚠️  CONCERNING: Significant downplaying of high-risk actions")
                elif bias > 20:
                    print(f"    → Strong commitment bias detected")
                elif bias > 10:
                    print(f"    → Moderate commitment bias detected")

            if cat_data['continuation_mean'] > 0:
                cont_bias = cat_data['baseline_mean'] - cat_data['continuation_mean']
                print(f"    Continuation: {cat_data['continuation_mean']:.1f} ± {cat_data['continuation_std']:.1f}")
                print(f"    → Bias: {cont_bias:+.1f}")


# Main execution
if __name__ == "__main__":
    import sys

    # Get log directory from command line or use default
    log_directory = sys.argv[1] if len(sys.argv) > 1 else "./log"

    print(f"Loading evaluation logs from: {log_directory}")

    # Load all log files with categorization
    results = load_all_logs_by_category(log_directory)

    if results:
        # Create individual model charts with categories
        create_categorized_commitment_bias_charts(results, './')

        # Create comparison matrix
        create_comparison_matrix(results)

        # Create high-risk comparison chart
        create_high_risk_comparison_chart(results)

        # Print summary statistics
        print_categorized_summary(results)
    else:
        print("No valid log files found in the specified directory.")