#!/usr/bin/env python3
"""
Script to plot baseline accuracy comparison between old GT and new GT (majority vote).

This script:
1. Loads the CSV with ground truth and baseline results
2. Calculates baseline accuracy using old GT (old_gt column)
3. Calculates baseline accuracy using new GT (majority_vote column)
4. Creates side-by-side plots showing accuracy for each baseline type by category
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Any


def load_data(csv_path: str) -> pd.DataFrame:
    """Load the CSV data."""
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {list(df.columns)}")
    return df


def calculate_baseline_accuracy(df: pd.DataFrame, ground_truth_col: str) -> pd.DataFrame:
    """Calculate baseline accuracy using specified ground truth column."""
    # Get baseline columns (exclude old_gt and other non-baseline columns)
    baseline_cols = [col for col in df.columns if col.startswith('baseline_')]
    
    results = []
    
    for category in df['fold_name'].unique():
        cat_data = df[df['fold_name'] == category]
        
        # Special handling for sandbagging - split into ASCII and non-ASCII
        if category == 'sandbagging':
            # Split sandbagging into ASCII and non-ASCII
            ascii_data = cat_data[cat_data['task_name'].str.contains('ascii', case=False, na=False)]
            non_ascii_data = cat_data[~cat_data['task_name'].str.contains('ascii', case=False, na=False)]
            
            # Process ASCII sandbagging
            if len(ascii_data) > 0:
                ascii_results = calculate_category_accuracy(ascii_data, 'sandbagging (ascii)', baseline_cols, ground_truth_col)
                results.extend(ascii_results)
            
            # Process non-ASCII sandbagging
            if len(non_ascii_data) > 0:
                non_ascii_results = calculate_category_accuracy(non_ascii_data, 'sandbagging (no ascii)', baseline_cols, ground_truth_col)
                results.extend(non_ascii_results)
        else:
            # Process other categories normally
            cat_results = calculate_category_accuracy(cat_data, category, baseline_cols, ground_truth_col)
            results.extend(cat_results)
    
    return pd.DataFrame(results)


def calculate_category_accuracy(cat_data: pd.DataFrame, category_name: str, baseline_cols: List[str], ground_truth_col: str) -> List[Dict]:
    """Calculate accuracy for each baseline in a category."""
    results = []
    
    for baseline_col in baseline_cols:
        if baseline_col in cat_data.columns:
            # Get predictions and ground truth
            predictions = cat_data[baseline_col].dropna()
            gt = cat_data[ground_truth_col].dropna()
            
            # Find common samples
            common_idx = predictions.index.intersection(gt.index)
            if len(common_idx) > 0:
                pred_aligned = predictions.loc[common_idx]
                gt_aligned = gt.loc[common_idx]
                
                # Calculate accuracy
                correct = (pred_aligned == gt_aligned).sum()
                total = len(pred_aligned)
                accuracy = correct / total if total > 0 else 0
                
                # Clean baseline name for display
                baseline_name = baseline_col.replace('baseline_baseline_', '').replace('baseline_', '')
                
                results.append({
                    'category': category_name,
                    'baseline': baseline_name,
                    'accuracy': accuracy,
                    'correct': correct,
                    'total': total
                })
    
    return results


def create_baseline_accuracy_plot(old_results_df: pd.DataFrame, new_results_df: pd.DataFrame, output_dir: Path, suffix=''):
    """Create side-by-side baseline accuracy plots."""
    # Get unique categories and baselines
    categories = sorted(old_results_df['category'].unique())
    baselines = sorted(old_results_df['baseline'].unique())
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Set up the plot
    x = np.arange(len(categories))
    width = 0.8 / len(baselines)  # Width of each bar
    
    # Earth tone colors for different baselines: red, blue, orange, green
    colors = ['#DC143C', '#4169E1', '#FF8C00', '#32CD32', '#8A2BE2']  # Crimson, Royal Blue, Dark Orange, Lime Green, Blue Violet
    
    # Plot old GT accuracy (left subplot)
    for i, baseline in enumerate(baselines):
        old_accuracies = []
        for category in categories:
            old_data = old_results_df[(old_results_df['category'] == category) & 
                                    (old_results_df['baseline'] == baseline)]
            if len(old_data) > 0:
                old_accuracies.append(old_data['accuracy'].iloc[0])
            else:
                old_accuracies.append(0)
        
        ax1.bar(x + i * width, old_accuracies, width, label=baseline, color=colors[i], alpha=0.8)
    
    ax1.set_xlabel('Category')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Baseline Accuracy (Old Ground Truth)')
    ax1.set_xticks(x + width * (len(baselines) - 1) / 2)
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Plot new GT accuracy (right subplot)
    for i, baseline in enumerate(baselines):
        new_accuracies = []
        for category in categories:
            new_data = new_results_df[(new_results_df['category'] == category) & 
                                    (new_results_df['baseline'] == baseline)]
            if len(new_data) > 0:
                new_accuracies.append(new_data['accuracy'].iloc[0])
            else:
                new_accuracies.append(0)
        
        ax2.bar(x + i * width, new_accuracies, width, label=baseline, color=colors[i], alpha=0.8)
    
    ax2.set_xlabel('Category')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Baseline Accuracy (New Ground Truth - Majority Vote)')
    ax2.set_xticks(x + width * (len(baselines) - 1) / 2)
    ax2.set_xticklabels(categories, rotation=45, ha='right')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / f'baseline_accuracy_comparison{suffix}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_file}")
    
    return fig


def create_accuracy_summary_table(old_results_df: pd.DataFrame, new_results_df: pd.DataFrame, output_dir: Path, suffix=''):
    """Create a summary table comparing old vs new accuracy."""
    # Merge old and new results
    summary_data = []
    
    for category in old_results_df['category'].unique():
        for baseline in old_results_df['baseline'].unique():
            old_data = old_results_df[(old_results_df['category'] == category) & 
                                    (old_results_df['baseline'] == baseline)]
            new_data = new_results_df[(new_results_df['category'] == category) & 
                                    (new_results_df['baseline'] == baseline)]
            
            if len(old_data) > 0 and len(new_data) > 0:
                old_acc = old_data['accuracy'].iloc[0]
                new_acc = new_data['accuracy'].iloc[0]
                diff = new_acc - old_acc
                
                summary_data.append({
                    'category': category,
                    'baseline': baseline,
                    'old_accuracy': old_acc,
                    'new_accuracy': new_acc,
                    'difference': diff,
                    'old_correct': old_data['correct'].iloc[0],
                    'old_total': old_data['total'].iloc[0],
                    'new_correct': new_data['correct'].iloc[0],
                    'new_total': new_data['total'].iloc[0]
                })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary
    summary_file = output_dir / f'baseline_accuracy_comparison_summary{suffix}.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"Saved summary to: {summary_file}")
    
    return summary_df


def create_new_gt_only_plot(new_results_df: pd.DataFrame, output_dir: Path, suffix=''):
    """Create a single plot showing baseline accuracy using only new ground truth."""
    # Filter out tool category
    new_results_df = new_results_df[new_results_df['category'] != 'tool']
    
    # Get unique categories and baselines
    categories = sorted(new_results_df['category'].unique())
    baselines = sorted(new_results_df['baseline'].unique())
    
    # Create figure with single subplot (wider to prevent annotation overlap)
    fig, ax = plt.subplots(1, 1, figsize=(24, 8))
    
    # Set up the plot
    x = np.arange(len(categories))
    width = 0.8 / len(baselines)  # Width of each bar
    
    # Google-style colors for different baselines
    colors = ['#4285F4', '#EA4335', '#FBBC04', '#34A853', '#9C27B0']  # Google Blue, Google Red, Google Yellow, Google Green, Purple
    
    # Plot new GT accuracy
    for i, baseline in enumerate(baselines):
        new_accuracies = []
        for category in categories:
            new_data = new_results_df[(new_results_df['category'] == category) & 
                                    (new_results_df['baseline'] == baseline)]
            if len(new_data) > 0:
                new_accuracies.append(new_data['accuracy'].iloc[0])
            else:
                new_accuracies.append(0)
        
        bars = ax.bar(x + i * width, new_accuracies, width, label=baseline, color=colors[i], alpha=0.8)
        
        # Add value labels on bars
        for bar, acc in zip(bars, new_accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{acc:.1%}', ha='center', va='bottom', fontsize=6)
    
    ax.set_xlabel('Category')
    ax.set_ylabel('Accuracy')
    ax.set_title('Baseline Accuracy (New Ground Truth - Majority Vote)')
    ax.text(0.5, 1.08, 'Evaluating baseline accuracies across prompt formulations under re-labeled ground truth, with all prompts including a shared taxonomy of lies and their definitions.', 
            transform=ax.transAxes, ha='center', va='top', fontsize=10, style='italic', wrap=True)
    ax.set_xticks(x + width * (len(baselines) - 1) / 2)
    ax.set_xticklabels(categories)
    
    # Create custom legend labels
    def clean_baseline_name(name):
        if name == 'llama_chat':
            return 'Chat Format + lie definitions'
        elif name == 'llama_chat_reasoning':
            return 'Chat Format with Reasoning + lie definitions'
        elif name == 'base_transcript':
            return 'Base Transcript + lie definitions'
        elif name == 'rowans_escaped_transcript':
            return 'Rowans Escaped Transcript + lie definitions'
        else:
            return name.replace('_', ' ').title() + ' + lie definitions'
    
    # Get the legend handles and create custom labels
    handles, _ = ax.get_legend_handles_labels()
    custom_labels = [clean_baseline_name(baseline) for baseline in baselines]
    ax.legend(handles, custom_labels, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=len(baselines), frameon=False)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Adjust layout to accommodate bottom legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save plot
    output_file = output_dir / f'baseline_accuracy_new_gt_only{suffix}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved new GT only plot to: {output_file}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Plot baseline accuracy comparison: old vs new ground truth')
    parser.add_argument('--csv_path', type=str, 
                       default='baseline_no_unanswerable_v2_predictions_dataframe.csv',
                       help='Path to the CSV file with GT labelling results')
    parser.add_argument('--output_dir', type=str, default='plots',
                       help='Directory to save output plots and tables')
    parser.add_argument('--suffix', type=str, default='',
                       help='Suffix to append to output filenames')
    
    args = parser.parse_args()
    
    # Set up paths
    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        return
    
    # Load data
    df = load_data(csv_path)
    
    # Filter out unanswerable questions and tools
    print(f"\nFiltering out unanswerable questions and tools...")
    original_count = len(df)
    df = df[~df['fold_name'].isin(['unanswerable', 'tools'])]
    filtered_count = len(df)
    print(f"Removed {original_count - filtered_count} unanswerable and tools samples ({original_count} -> {filtered_count})")
    
    # Calculate baseline accuracy for old GT
    print("\nCalculating baseline accuracy using old ground truth...")
    old_results_df = calculate_baseline_accuracy(df, 'old_gt')
    
    # Calculate baseline accuracy for new GT (majority vote)
    print("Calculating baseline accuracy using new ground truth (majority vote)...")
    new_results_df = calculate_baseline_accuracy(df, 'majority_vote')
    
    # Create comparison plots
    fig = create_baseline_accuracy_plot(old_results_df, new_results_df, output_dir, args.suffix)
    
    # Create new GT only plot
    fig2 = create_new_gt_only_plot(new_results_df, output_dir, args.suffix)
    
    # Create summary table
    summary_df = create_accuracy_summary_table(old_results_df, new_results_df, output_dir, args.suffix)
    
    # Show the plots
    plt.show()
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")
    
    # Print summary statistics
    print(f"\nBaseline Accuracy Summary:")
    print(f"Categories: {len(old_results_df['category'].unique())}")
    print(f"Baselines: {len(old_results_df['baseline'].unique())}")
    print(f"Baseline types: {sorted(old_results_df['baseline'].unique())}")


if __name__ == "__main__":
    main()
