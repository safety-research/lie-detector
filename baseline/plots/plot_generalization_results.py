#!/usr/bin/env python3
"""
Plot generalization results as histograms with error bars.
Shows results for both map_1 and map_2, with three baseline types per category.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from pathlib import Path
import glob

def load_generalization_results():
    """Load the generalization CSV results from individual baseline folders."""
    results_dir = "results"
    all_data = []
    
    # Baseline types and their folders
    baseline_types = {
        'base_transcript': 'base_transcript',
        'escaped_transcript': 'escaped_transcript', 
        'llama_chat': 'llama_chat',
        'llama_chat_reasoning': 'llama_chat_reasoning',
        'base_transcript_reasoning': 'base_transcript_reasoning',
        'rowans_escaped_transcript': 'rowans_escaped_transcript'
    }
    
    for baseline_type, folder in baseline_types.items():
        folder_path = os.path.join(results_dir, folder)
        if not os.path.exists(folder_path):
            continue
            
        # Find generalization CSV files for this baseline type
        pattern = os.path.join(folder_path, "generalization_metrics_*.csv")
        csv_files = glob.glob(pattern)
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                # Extract model name from filename
                filename = os.path.basename(csv_file)
                if 'llama_3.1_8b' in filename:
                    model = 'meta-llama_llama-3.1-8b-instruct'
                elif 'llama_3.3_70b' in filename:
                    model = 'meta-llama_llama-3.3-70b-instruct'
                else:
                    continue
                
                # Add baseline type and model columns
                df['baseline_type'] = baseline_type
                df['model'] = model
                
                all_data.append(df)
                
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
    
    if not all_data:
        print("No generalization CSV files found in results directory")
        return None
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def create_generalization_plots(df):
    """Create histograms for generalization results."""
    # Filter for llama 8b model
    llama_8b_df = df[df['model'] == 'meta-llama_llama-3.1-8b-instruct'].copy()
    
    if llama_8b_df.empty:
        print("No data found for llama-3.1-8b-instruct model")
        return
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))  # Increased height for legend
    
    # Add overall title
    fig.suptitle('Precision for Zero Shot Baselines Llama 8b Instruct', fontsize=16, fontweight='bold')
    
    # Plot for map_1
    plot_category_results(ax1, llama_8b_df, 'map_1', 'Map 1: External Influence & Drive Categories')
    
    # Plot for map_2
    plot_category_results(ax2, llama_8b_df, 'map_2', 'Map 2: Internal vs External Knowledge Categories')
    
    # Create a single legend at the bottom
    baseline_labels = ['Base Transcript', 'Escaped Transcript', 'Llama Chat', 'Llama Chat Reasoning', 'Base Transcript Reasoning', 'Rowans Escaped Transcript']
    colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948"]  # Updated color palette
    
    # Create legend handles
    legend_elements = []
    for label, color in zip(baseline_labels, colors):
        legend_elements.append(patches.Rectangle((0,0),1,1, facecolor=color, label=label))
    
    # Add legend at the bottom outside the plots
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=3)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Further reduced bottom margin to move legend higher
    plt.savefig('results/generalization_plots.png', dpi=300, bbox_inches='tight')
    # plt.show()  # Removed to prevent pop-ups
    print("Plots saved to results/generalization_plots.png")

def plot_category_results(ax, df, map_prefix, title):
    """Plot results for a specific map prefix ('map_1' or 'map_2')."""
    # Use 'Dataset / Task' as the category column
    category_col = 'Dataset / Task'
    # Get unique categories for this map prefix (excluding 'OVERALL')
    categories = [cat for cat in df[category_col].unique() if cat.startswith(map_prefix)]
    categories = sorted(categories)
    
    # Clean up category labels for display
    display_labels = []
    for category in categories:
        # Remove the map prefix (e.g., "map_1_instruction_following" -> "instruction_following")
        clean_label = category.replace(f'{map_prefix}_', '')
        # Replace underscores with spaces
        clean_label = clean_label.replace('_', ' ')
        # Capitalize first letter of each word
        clean_label = ' '.join(word.capitalize() for word in clean_label.split())
        display_labels.append(clean_label)
    
    # Baseline types
    baseline_types = ['base_transcript', 'escaped_transcript', 'llama_chat', 'llama_chat_reasoning', 'base_transcript_reasoning', 'rowans_escaped_transcript']
    
    # Colors for baseline types
    colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948"]  # Updated color palette
    
    # Set up the plot
    x = np.arange(len(categories))
    width = 0.13  # Reduced width to accommodate 6 bars
    
    # Plot bars for each baseline type
    for i, (baseline_type, color) in enumerate(zip(baseline_types, colors)):
        values = []
        errors = []
        
        for category in categories:
            # Get data for this category and baseline type
            mask = (df[category_col] == category) & (df['baseline_type'] == baseline_type)
            category_data = df[mask]
            
            if not category_data.empty:
                precision = category_data['precision'].iloc[0]
                std = category_data['precision_se'].iloc[0] if 'precision_se' in category_data.columns else 0
                values.append(precision)
                errors.append(std)
            else:
                values.append(0)
                errors.append(0)
        
        # Plot bars with error bars
        bars = ax.bar(x + i * width, values, width, color=color)
        
        # Add error bars if we have standard error data
        if any(err != 0 for err in errors):
            ax.errorbar(x + i * width, values, yerr=errors, fmt='none', color='black', capsize=3, capthick=1)
            print(f"Added error bars for {baseline_type}: {errors}")
        else:
            print(f"No error bars for {baseline_type}: all errors are 0")
        
        # Add value labels on bars (positioned above error bars)
        for j, (bar, value, error) in enumerate(zip(bars, values, errors)):
            height = bar.get_height()
            # Position text above the error bar if there is one, otherwise above the bar
            text_y = height + error + 0.02 if error > 0 else height + 0.02
            ax.text(bar.get_x() + bar.get_width()/2., text_y,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=6)  # Reduced font size
    
    # Customize the plot
    ax.set_xlabel('Categories')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.set_xticks(x + 2.5 * width)  # Center the x-ticks between the 6 bars
    ax.set_xticklabels(display_labels, ha='center')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)  # Precision is between 0 and 1, extend just slightly above 1

def main():
    """Main function to create generalization plots."""
    print("Loading generalization results...")
    df = load_generalization_results()
    
    if df is None:
        return
    
    print("Creating generalization plots...")
    create_generalization_plots(df)
    print("Done!")

if __name__ == "__main__":
    main() 