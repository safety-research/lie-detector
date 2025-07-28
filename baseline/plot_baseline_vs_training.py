#!/usr/bin/env python3
"""
Plot baseline results vs training results for Llama 8B Instruct.
Shows precision comparisons across different baseline types and training formats.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import numpy as np
import os
import json
from pathlib import Path
import glob

def load_baseline_json_results():
    """Load baseline results from JSON files."""
    baseline_data = []
    
    # Baseline JSON files
    baseline_files = {
        'base_transcript': 'results/base_transcript/results_2025-07-16_07-35-56_meta-llama_llama-3.1-8b-instruct_base_transcript.json',
        'escaped_transcript': 'results/escaped_transcript/results_2025-07-16_08-03-54_meta-llama_llama-3.1-8b-instruct_escaped_transcript.json',
        'llama_chat': 'results/llama_chat/results_2025-07-16_08-18-56_meta-llama_llama-3.1-8b-instruct_llama_chat.json'
    }
    
    for baseline_type, file_path in baseline_files.items():
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found")
            continue
            
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract overall metrics
            overall_metrics = data.get('overall_metrics', {})
            precision = overall_metrics.get('precision', 0)
            
            # Calculate standard error for precision
            # SE = sqrt(precision * (1 - precision) / n)
            parseable_samples = overall_metrics.get('parseable_samples', 0)
            if parseable_samples > 0:
                precision_se = np.sqrt(precision * (1 - precision) / parseable_samples)
            else:
                precision_se = 0
            
            baseline_row = {
                'type': baseline_type,
                'precision': precision,
                'precision_se': precision_se,
                'parseable_samples': parseable_samples,
                'category': 'baseline'
            }
            
            baseline_data.append(baseline_row)
            print(f"Loaded {baseline_type}: precision = {precision:.4f} ± {precision_se:.4f}")
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if not baseline_data:
        print("No baseline JSON files found")
        return None
    
    return pd.DataFrame(baseline_data)

def load_training_results():
    """Load training results from train/results directory."""
    training_data = []
    
    # Training result files
    training_files = {
        'base_transcript_sft': '../train/results/base_transcript_sft.json',
        'llama_chat_sft': '../train/results/llama_chat_sft.json'
    }
    
    for training_type, file_path in training_files.items():
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found")
            continue
            
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract validation metrics
            val_metrics = data.get('validation_metrics', {})
            
            # Calculate overall precision (using macro precision as it's balanced)
            macro_precision = val_metrics.get('macro_precision', 0)
            
            # For training, we don't have individual sample data to calculate SE
            # We'll use 0 for now, or could estimate from validation set size
            total_samples = val_metrics.get('total_samples', 0)
            if total_samples > 0:
                # Rough estimate of SE based on validation set size
                precision_se = np.sqrt(macro_precision * (1 - macro_precision) / total_samples)
            else:
                precision_se = 0
            
            # Create a row for this training result
            training_row = {
                'type': training_type.replace('_sft', ''),
                'precision': macro_precision,
                'precision_se': precision_se,
                'parseable_samples': total_samples,
                'category': 'training'
            }
            
            training_data.append(training_row)
            print(f"Loaded {training_type}: precision = {macro_precision:.4f} ± {precision_se:.4f}")
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if not training_data:
        print("No training results found")
        return None
    
    return pd.DataFrame(training_data)

def create_comprehensive_plot(baseline_df, training_df):
    """Create a comprehensive plot with all baseline and training results."""
    if baseline_df is None and training_df is None:
        print("No data found!")
        return
    
    # Combine baseline and training data
    all_data = []
    if baseline_df is not None:
        all_data.append(baseline_df)
    if training_df is not None:
        all_data.append(training_df)
    
    if not all_data:
        print("No data to plot!")
        return
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    # Add title
    fig.suptitle('Baseline vs Training Results on Llama 8B Instruct', fontsize=16, fontweight='bold')
    
    # Define the categories we want to show
    # For each data format, we want to show all baseline types + training
    categories = []
    precisions = []
    errors = []
    colors = []
    labels = []
    
    # Define baseline types and their colors
    baseline_types = ['base_transcript', 'escaped_transcript', 'llama_chat']
    baseline_colors = ['#1f77b4', '#2ca02c', '#d62728']  # Blue, Green, Red
    
    # For each baseline type, show baseline results
    for i, baseline_type in enumerate(baseline_types):
        # Get baseline data for this type
        baseline_data = combined_df[(combined_df['type'] == baseline_type) & (combined_df['category'] == 'baseline')]
        if not baseline_data.empty:
            prec = baseline_data['precision'].iloc[0]
            err = baseline_data['precision_se'].iloc[0]
            
            categories.append(f"{baseline_type.replace('_', ' ').title()}")
            precisions.append(prec)
            errors.append(err)
            colors.append(baseline_colors[i])
            labels.append('Baseline')
    
    # For training, show base_transcript and llama_chat results
    training_types = ['base_transcript', 'llama_chat']
    training_colors = ['#ff7f0e', '#9467bd']  # Orange, Purple
    
    for i, training_type in enumerate(training_types):
        # Get training data for this type
        training_data = combined_df[(combined_df['type'] == training_type) & (combined_df['category'] == 'training')]
        if not training_data.empty:
            prec = training_data['precision'].iloc[0]
            err = training_data['precision_se'].iloc[0]
            
            categories.append(f"{training_type.replace('_', ' ').title()} (SFT)")
            precisions.append(prec)
            errors.append(err)
            colors.append(training_colors[i])
            labels.append('Training')
    
    if not categories:
        print("No valid data to plot!")
        return
    
    # Set up the plot
    x = np.arange(len(categories))
    width = 0.7
    
    # Plot bars
    bars = ax.bar(x, precisions, width, color=colors, alpha=0.8)
    
    # Add error bars for ALL bars
    ax.errorbar(x, precisions, yerr=errors, fmt='none', color='black', capsize=3, capthick=1)
    
    # Add value labels on bars
    for i, (bar, value, error) in enumerate(zip(bars, precisions, errors)):
        height = bar.get_height()
        # Position text above the error bar
        text_y = height + error + 0.02 if error > 0 else height + 0.02
        ax.text(bar.get_x() + bar.get_width()/2., text_y,
               f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Create custom legend
    legend_elements = []
    legend_labels = []
    
    # Add baseline legend entries
    for i, baseline_type in enumerate(baseline_types):
        legend_elements.append(Rectangle((0,0),1,1, facecolor=baseline_colors[i], alpha=0.8))
        legend_labels.append(f"{baseline_type.replace('_', ' ').title()} (Baseline)")
    
    # Add training legend entries
    for i, training_type in enumerate(training_types):
        legend_elements.append(Rectangle((0,0),1,1, facecolor=training_colors[i], alpha=0.8))
        legend_labels.append(f"{training_type.replace('_', ' ').title()} (Training)")
    
    # Customize the plot
    ax.set_xlabel('Data Format', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision Comparison: All Baseline and Training Results', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend(legend_elements, legend_labels, fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # Add some spacing for rotated labels
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('results/baseline_vs_training_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Plot saved to results/baseline_vs_training_comprehensive.png")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY OF RESULTS")
    print("="*60)
    for i, category in enumerate(categories):
        print(f"{category}: {precisions[i]:.4f} ± {errors[i]:.4f}")
    print()
    
    # Print improvements
    print("IMPROVEMENTS (Training - Baseline):")
    print("-" * 40)
    for training_type in training_types:
        # Find baseline result
        baseline_idx = None
        for i, cat in enumerate(categories):
            if cat.lower().startswith(training_type.replace('_', ' ')) and 'sft' not in cat.lower():
                baseline_idx = i
                break
        
        # Find training result
        training_idx = None
        for i, cat in enumerate(categories):
            if cat.lower().startswith(training_type.replace('_', ' ')) and 'sft' in cat.lower():
                training_idx = i
                break
        
        if baseline_idx is not None and training_idx is not None:
            improvement = precisions[training_idx] - precisions[baseline_idx]
            print(f"{training_type.replace('_', ' ').title()}: {improvement:+.4f}")

def main():
    """Main function to create comparison plots."""
    print("Loading baseline JSON results...")
    baseline_df = load_baseline_json_results()
    
    print("\nLoading training results...")
    training_df = load_training_results()
    
    print("\nCreating comprehensive comparison plot...")
    create_comprehensive_plot(baseline_df, training_df)
    print("Done!")

if __name__ == "__main__":
    main() 