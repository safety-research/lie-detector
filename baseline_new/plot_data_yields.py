#!/usr/bin/env python3
"""
Plot data yields per fold based on majority vote from GT labelling results.
This script analyzes the percentage of samples where the model lied (majority vote = 'A')
across different folds/categories.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse

def load_and_analyze_data(csv_path):
    """Load the CSV data and analyze majority vote patterns."""
    print(f"Loading data from: {csv_path}")
    
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {list(df.columns)}")
    
    # Check the unique values in majority_vote column
    print(f"Unique majority_vote values: {df['majority_vote'].unique()}")
    print(f"Majority vote value counts:")
    print(df['majority_vote'].value_counts())
    
    return df

def calculate_data_yields_per_fold(df, use_old_gt=False):
    """Calculate data yields (percentage of lies) per fold."""
    results = []
    
    # Filter out tool category
    df = df[df['fold_name'] != 'tool']
    
    for fold_name in df['fold_name'].unique():
        fold_data = df[df['fold_name'] == fold_name]
        
        # Special handling for sandbagging - split into ASCII and non-ASCII
        if fold_name == 'sandbagging':
            # Split sandbagging into ASCII and non-ASCII
            ascii_data = fold_data[fold_data['task_name'].str.contains('ascii', case=False, na=False)]
            non_ascii_data = fold_data[~fold_data['task_name'].str.contains('ascii', case=False, na=False)]
            
            # Process ASCII sandbagging
            if len(ascii_data) > 0:
                ascii_result = process_fold_data(ascii_data, 'sandbagging (ascii)', use_old_gt)
                results.append(ascii_result)
            
            # Process non-ASCII sandbagging
            if len(non_ascii_data) > 0:
                non_ascii_result = process_fold_data(non_ascii_data, 'sandbagging (no ascii)', use_old_gt)
                results.append(non_ascii_result)
        else:
            # Process other folds normally
            fold_result = process_fold_data(fold_data, fold_name, use_old_gt)
            results.append(fold_result)
    
    return pd.DataFrame(results)

def process_fold_data(fold_data, fold_name, use_old_gt):
    """Process data for a single fold and return results dictionary."""
    total_samples = len(fold_data)
    
    if use_old_gt:
        # Use old ground truth from baseline data
        if 'old_gt' in fold_data.columns:
            lie_samples = len(fold_data[fold_data['old_gt'] == 'A'])
            truth_samples = len(fold_data[fold_data['old_gt'] == 'B'])
            other_samples = total_samples - lie_samples - truth_samples
        else:
            # Fallback to majority vote if no old_gt column
            lie_samples = len(fold_data[fold_data['majority_vote'] == 'A'])
            truth_samples = len(fold_data[fold_data['majority_vote'] == 'B'])
            other_samples = total_samples - lie_samples - truth_samples
    else:
        # Use majority vote (new data yield)
        lie_samples = len(fold_data[fold_data['majority_vote'] == 'A'])
        truth_samples = len(fold_data[fold_data['majority_vote'] == 'B'])
        other_samples = total_samples - lie_samples - truth_samples
    
    # Calculate percentages
    lie_percentage = (lie_samples / total_samples) * 100 if total_samples > 0 else 0
    truth_percentage = (truth_samples / total_samples) * 100 if total_samples > 0 else 0
    other_percentage = (other_samples / total_samples) * 100 if total_samples > 0 else 0
    
    return {
        'fold_name': fold_name,
        'total_samples': total_samples,
        'lie_samples': lie_samples,
        'truth_samples': truth_samples,
        'other_samples': other_samples,
        'lie_percentage': lie_percentage,
        'truth_percentage': truth_percentage,
        'other_percentage': other_percentage
    }

def create_data_yields_plot(old_results_df, new_results_df, output_dir, suffix=''):
    """Create side-by-side horizontal bar charts comparing old vs new data yields."""
    
    # Set up the plotting style
    plt.style.use('default')
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Sort results alphabetically by fold_name for consistent y-axis labels
    old_results_sorted = old_results_df.sort_values('fold_name', ascending=True)
    new_results_sorted = new_results_df.sort_values('fold_name', ascending=True)
    
    # Create horizontal bar plots
    bars1 = ax1.barh(old_results_sorted['fold_name'], old_results_sorted['lie_percentage'], 
                     color='lightcoral', alpha=0.7, edgecolor='darkred', linewidth=1)
    bars2 = ax2.barh(new_results_sorted['fold_name'], new_results_sorted['lie_percentage'], 
                     color='lightblue', alpha=0.7, edgecolor='darkblue', linewidth=1)
    
    # Configure left subplot (old data yield)
    ax1.set_xlabel('Percentage of Lies (%)', fontsize=12)
    ax1.set_title('Old Data Yield\n(Using Original Ground Truth)', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Configure right subplot (new data yield)
    ax2.set_xlabel('Percentage of Lies (%)', fontsize=12)
    ax2.set_title('New Data Yield\n(Using Majority Vote)', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars1, old_results_sorted['lie_percentage'])):
        ax1.text(value + 1, bar.get_y() + bar.get_height()/2, 
                f'{value:.1f}%', va='center', fontsize=9, fontweight='bold')
    
    for i, (bar, value) in enumerate(zip(bars2, new_results_sorted['lie_percentage'])):
        ax2.text(value + 1, bar.get_y() + bar.get_height()/2, 
                f'{value:.1f}%', va='center', fontsize=9, fontweight='bold')
    
    # Set x-axis limits to show all values clearly
    max_old = max(old_results_sorted['lie_percentage']) if len(old_results_sorted) > 0 else 0
    max_new = max(new_results_sorted['lie_percentage']) if len(new_results_sorted) > 0 else 0
    max_val = max(max_old, max_new)
    
    ax1.set_xlim(0, max_val * 1.1)
    ax2.set_xlim(0, max_val * 1.1)
    
    # Add overall title
    fig.suptitle('Data Yields Comparison: Old vs New Ground Truth', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = output_dir / f'data_yields_comparison{suffix}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")
    
    return fig

def create_summary_table(old_results_df, new_results_df, output_dir, suffix=''):
    """Create a summary table comparing old vs new results."""
    
    # Merge the two dataframes for comparison
    comparison_df = old_results_df.merge(new_results_df, on='fold_name', suffixes=('_old', '_new'))
    
    # Calculate differences
    comparison_df['lie_percentage_diff'] = comparison_df['lie_percentage_new'] - comparison_df['lie_percentage_old']
    comparison_df['lie_samples_diff'] = comparison_df['lie_samples_new'] - comparison_df['lie_samples_old']
    
    # Sort by difference in lie percentage
    comparison_df = comparison_df.sort_values('lie_percentage_diff', ascending=False)
    
    # Save as CSV
    summary_path = output_dir / f'data_yields_comparison_summary{suffix}.csv'
    comparison_df.to_csv(summary_path, index=False)
    print(f"Saved comparison summary to: {summary_path}")
    
    # Print summary to console
    print("\n" + "="*120)
    print("DATA YIELDS COMPARISON SUMMARY")
    print("="*120)
    print(f"{'Fold Name':<20} {'Old Lie %':<10} {'New Lie %':<10} {'Difference':<12} {'Old Lies':<8} {'New Lies':<8} {'Diff':<6}")
    print("-"*120)
    
    for _, row in comparison_df.iterrows():
        print(f"{row['fold_name']:<20} {row['lie_percentage_old']:<10.1f} {row['lie_percentage_new']:<10.1f} "
              f"{row['lie_percentage_diff']:<+12.1f} {row['lie_samples_old']:<8} {row['lie_samples_new']:<8} "
              f"{row['lie_samples_diff']:<+6}")
    
    print("-"*120)
    total_old_lies = comparison_df['lie_samples_old'].sum()
    total_new_lies = comparison_df['lie_samples_new'].sum()
    total_samples = comparison_df['total_samples_old'].sum()
    old_percentage = (total_old_lies / total_samples) * 100
    new_percentage = (total_new_lies / total_samples) * 100
    diff_percentage = new_percentage - old_percentage
    
    print(f"{'TOTAL':<20} {old_percentage:<10.1f} {new_percentage:<10.1f} "
          f"{diff_percentage:<+12.1f} {total_old_lies:<8} {total_new_lies:<8} "
          f"{total_new_lies - total_old_lies:<+6}")
    
    return comparison_df

def main():
    parser = argparse.ArgumentParser(description='Plot data yields comparison: old vs new ground truth')
    parser.add_argument('--csv_path', type=str, 
                       default='baseline_final_improved_predictions_dataframe.csv',
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
    
    # Load and analyze data
    df = load_and_analyze_data(csv_path)
    
    # Filter out unanswerable questions and tools
    print(f"\nFiltering out unanswerable questions and tools...")
    original_count = len(df)
    df = df[~df['fold_name'].isin(['unanswerable', 'tools'])]
    filtered_count = len(df)
    print(f"Removed {original_count - filtered_count} unanswerable and tools samples ({original_count} -> {filtered_count})")
    
    # Calculate data yields per fold for both old and new ground truth
    print("\nCalculating old data yields (using original ground truth)...")
    old_results_df = calculate_data_yields_per_fold(df, use_old_gt=True)
    
    print("Calculating new data yields (using majority vote)...")
    new_results_df = calculate_data_yields_per_fold(df, use_old_gt=False)
    
    # Create comparison plots
    fig = create_data_yields_plot(old_results_df, new_results_df, output_dir, args.suffix)
    
    # Create comparison summary table
    summary_df = create_summary_table(old_results_df, new_results_df, output_dir, args.suffix)
    
    # Show the plot
    plt.show()
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")

if __name__ == "__main__":
    main()
