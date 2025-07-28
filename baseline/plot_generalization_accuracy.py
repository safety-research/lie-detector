#!/usr/bin/env python3
"""
Plot accuracy across model sizes for different generalization mappings and baseline types.
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

# Configuration
RESULTS_DIR = "/Users/dipikakhullar/Desktop/lie-detector/baseline/organized_evaluation_20250721_215718/balanced_results"
TAXONOMY_FILE = "/Users/dipikakhullar/Desktop/lie-detector/baseline/lie_taxonomy_clean3.csv"

# Model size mapping
MODEL_SIZE_MAPPING = {
    'openrouter/google/gemma-3-4b-it': '4b',
    'openrouter/google/gemma-3-12b-it': '12b',
    'openrouter/google/gemma-3-27b-it': '27b'
}

# Baseline types from the results folder
BASELINE_TYPES = ['base_transcript', 'llama_chat', 'llama_chat_reasoning', 'rowans_escaped_transcript']

# Mapping for display names
BASELINE_DISPLAY_NAMES = {
    'base_transcript': 'base_transcript',
    'llama_chat': 'chat',
    'llama_chat_reasoning': 'chat_reasoning',
    'rowans_escaped_transcript': 'rowans_escaped_transcript'
}

def build_generalization_maps(df: pd.DataFrame, map_columns):
    """Build generalization mappings from CSV for each specified column."""
    mappings = {}
    for map_col in map_columns:
        mapping = defaultdict(list)
        for _, row in df.iterrows():
            task = str(row['Task']).strip().lower()
            category = str(row[map_col]).strip().lower()
            if task and category and category != 'nan':
                # Handle "Falsification / Exaggeration" by adding to both categories
                if '/' in category:
                    categories = [cat.strip() for cat in category.split('/')]
                    for cat in categories:
                        mapping[cat].append(task.replace('_', '-'))
                else:
                mapping[category].append(task.replace('_', '-'))
        mappings[map_col] = dict(mapping)
    return mappings

def load_taxonomy_mappings():
    """Load taxonomy and build generalization mappings."""
    df = pd.read_csv(TAXONOMY_FILE)
    map_columns = [col for col in df.columns if col.strip().lower() != 'task']
    return build_generalization_maps(df, map_columns)

def load_results_data():
    """Load all results data from the results directory."""
    results_data = {}
    
    for baseline_type in BASELINE_TYPES:
        baseline_dir = Path(RESULTS_DIR) / baseline_type
        if not baseline_dir.exists():
            print(f"Warning: {baseline_dir} does not exist")
            continue
            
        results_data[baseline_type] = {}
        
        # Find all JSON result files for this baseline type
        json_files = list(baseline_dir.glob("*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Extract model from metadata
                model = data.get('metadata', {}).get('model', 'unknown')
                model_size = MODEL_SIZE_MAPPING.get(model, 'unknown')
                
                if model_size == 'unknown':
                    print(f"Warning: Unknown model {model} in {json_file}")
                    continue
                
                # Get overall accuracy
                accuracy = data.get('overall_metrics', {}).get('accuracy', 0.0)
                
                # Get subtask accuracies
                subtask_accuracies = {}
                for subtask_name, subtask_data in data.get('subtask_results', {}).items():
                    if 'metrics' in subtask_data:
                        subtask_accuracies[subtask_name] = subtask_data['metrics'].get('accuracy', 0.0)
                
                results_data[baseline_type][model_size] = {
                    'overall_accuracy': accuracy,
                    'subtask_accuracies': subtask_accuracies
                }
                
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                continue
    
    return results_data

def calculate_category_accuracy(subtask_accuracies: Dict[str, float], tasks_in_category: List[str]) -> float:
    """Calculate average accuracy for tasks in a category."""
    if not tasks_in_category:
        return 0.0
    
    accuracies = []
    for task in tasks_in_category:
        # Convert task name format (e.g., "ascii-sandbagging-task" -> "ascii_sandbagging_task")
        task_key = task.replace('-', '_')
        if task_key in subtask_accuracies:
            accuracies.append(subtask_accuracies[task_key])
    
    if not accuracies:
        return 0.0
    
    return sum(accuracies) / len(accuracies)

def create_generalization_accuracy_plots(generalization_maps: Dict[str, Dict[str, List[str]]], 
                                       results_data: Dict[str, Dict[str, Any]]):
    """Create line plots for each generalization mapping."""
    model_sizes = ['4b', '12b', '27b']
    colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F"]
    
    # Create output directory
    os.makedirs('baseline_plots', exist_ok=True)
    
    # For each generalization mapping (row)
    for map_name, categories in generalization_maps.items():
        print(f"Creating plots for {map_name}...")
        
        # Determine grid layout for this mapping
        num_categories = len(categories)
        if num_categories <= 3:
            cols = num_categories
        else:
            cols = 3
        
        rows = (num_categories + cols - 1) // cols  # Ceiling division
        
        # Create subplot grid
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        fig.suptitle(f'{map_name}: Accuracy Across Model Sizes', fontsize=16, fontweight='bold')
        
        # For each category in this mapping
        for i, (category, tasks) in enumerate(categories.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # For each baseline type
            for j, baseline_type in enumerate(BASELINE_TYPES):
                if baseline_type not in results_data:
                    continue
                
                accuracies = []
                for model_size in model_sizes:
                    if model_size in results_data[baseline_type]:
                        subtask_accuracies = results_data[baseline_type][model_size]['subtask_accuracies']
                        category_accuracy = calculate_category_accuracy(subtask_accuracies, tasks)
                        accuracies.append(category_accuracy)
                    else:
                        accuracies.append(0.0)
                
                # Plot line for this baseline type
                ax.plot(model_sizes, accuracies, 
                       color=colors[j % len(colors)], 
                       marker='o', 
                       linewidth=2, 
                       markersize=6,
                       label=BASELINE_DISPLAY_NAMES[baseline_type].replace('_', ' ').title())
                
                # Add value labels
                for k, accuracy in enumerate(accuracies):
                    if accuracy > 0:
                        ax.annotate(f'{accuracy:.3f}', 
                                  xy=(k, accuracy), 
                                  xytext=(0, 5),
                                  textcoords='offset points', 
                                  ha='center', va='bottom',
                                  fontsize=8, fontweight='bold')
            
            ax.set_title(f'{category.title()}', fontweight='bold')
            ax.set_ylabel('Accuracy')
            ax.set_xlabel('Model Size')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # Hide unused subplots
        for i in range(len(categories), len(axes)):
            axes[i].set_visible(False)
        
        # Adjust layout
        plt.tight_layout(rect=(0, 0, 0.85, 0.95))
        
        # Save plot
        output_path = f"baseline_plots/{map_name.replace(' ', '_').lower()}_accuracy.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()



def print_summary_stats(generalization_maps: Dict[str, Dict[str, List[str]]], 
                       results_data: Dict[str, Dict[str, Any]]):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("ACCURACY SUMMARY BY GENERALIZATION MAPPING")
    print("="*80)
    
    model_sizes = ['4b', '12b', '27b']
    
    for map_name, categories in generalization_maps.items():
        print(f"\n{map_name}:")
        print("-" * 50)
        
        for category, tasks in categories.items():
            print(f"  {category}:")
            
            for baseline_type in BASELINE_TYPES:
                if baseline_type not in results_data:
                    continue
                
                accuracies = []
                for model_size in model_sizes:
                    if model_size in results_data[baseline_type]:
                        subtask_accuracies = results_data[baseline_type][model_size]['subtask_accuracies']
                        category_accuracy = calculate_category_accuracy(subtask_accuracies, tasks)
                        accuracies.append(category_accuracy)
                    else:
                        accuracies.append(0.0)
                
                print(f"    {BASELINE_DISPLAY_NAMES[baseline_type]}: {accuracies[0]:.3f}, {accuracies[1]:.3f}, {accuracies[2]:.3f}")





def create_combined_plot(generalization_maps: Dict[str, Dict[str, List[str]]], 
                        results_data: Dict[str, Dict[str, Any]]):
    """Create one combined plot with all generalization mappings."""
    model_sizes = ['4b', '12b', '27b']
    colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F"]
    
    # Determine grid size
    num_mappings = len(generalization_maps)
    max_categories = max(len(categories) for categories in generalization_maps.values())
    
    # Create grid
    fig, axes = plt.subplots(num_mappings, max_categories, 
                            figsize=(5*max_categories, 4*num_mappings),
                            squeeze=False)
    # plt.subplots_adjust(hspace=1.0, wspace=0.3, top=0.85, bottom=0.1)  # Increased hspace, lowered top slightly

    
    plt.subplots_adjust(hspace=0.6, wspace=0.3, top=0.90, bottom=0.1)

    # Main title
    fig.suptitle('Accuracy Across Model Sizes by Generalization Mapping', 
                 fontsize=18, fontweight='bold', y=0.99)  # Raised main title slightly

    # For each generalization mapping (row)
    for row_idx, (map_name, categories) in enumerate(generalization_maps.items()):
        categories_list = list(categories.items())

        # Calculate proper vertical position for each row title
        subplot_pos = axes[row_idx][0].get_position()
        title_y = subplot_pos.y1 + 0.02  # Lowered row title slightly

        fig.text(0.5, title_y, map_name,
                 ha='center', va='center', fontsize=14, fontweight='bold')
        
        # For each category (column)
        for col_idx in range(max_categories):
            ax = axes[row_idx][col_idx]
            
            if col_idx < len(categories_list):
                category, tasks = categories_list[col_idx]
                
                # For each baseline type
                for j, baseline_type in enumerate(BASELINE_TYPES):
                    if baseline_type not in results_data:
                        continue
                    
                    accuracies = []
                    for model_size in model_sizes:
                        if model_size in results_data[baseline_type]:
                            subtask_accuracies = results_data[baseline_type][model_size]['subtask_accuracies']
                            category_accuracy = calculate_category_accuracy(subtask_accuracies, tasks)
                            accuracies.append(category_accuracy)
                        else:
                            accuracies.append(0.0)
                    
                    # Plot line for this baseline type
                    ax.plot(model_sizes, accuracies, 
                           color=colors[j % len(colors)], 
                           marker='o', 
                           linewidth=2, 
                           markersize=6,
                           label=BASELINE_DISPLAY_NAMES[baseline_type].replace('_', ' ').title())
                
                ax.set_title(category.replace('_', ' ').title(),
                             fontsize=10, fontweight='bold')
                ax.set_ylim(0, 1)
                ax.set_ylabel('Accuracy')
                ax.set_xlabel('Model Size')
                ax.grid(True, alpha=0.3)
            else:
                ax.axis('off')
    
    # Create legend handles and labels for the main legend
    legend_handles = []
    legend_labels = []
    for j, baseline_type in enumerate(BASELINE_TYPES):
        if baseline_type in results_data:
            dummy_line, = plt.plot([], [], color=colors[j % len(colors)], 
                                   marker='o', linewidth=2, markersize=6, 
                                   label=BASELINE_DISPLAY_NAMES[baseline_type].replace('_', ' ').title())
            legend_handles.append(dummy_line)
            legend_labels.append(BASELINE_DISPLAY_NAMES[baseline_type].replace('_', ' ').title())
    
    # Add legend directly under the main title
    # fig.legend(handles=legend_handles, labels=legend_labels, 
    #            loc='upper center', bbox_to_anchor=(0.5, 0.91),  # Slightly below main title
    #            ncol=len(legend_labels), fontsize=10, frameon=True)

    fig.legend(handles=legend_handles, labels=legend_labels, 
           loc='upper center', bbox_to_anchor=(0.5, 0.965),  # Directly below title
           ncol=len(legend_labels), fontsize=10, frameon=True)

    
    # plt.tight_layout(rect=(0, 0, 1, 0.88))  # Adjusted rect for spacing

    os.makedirs('baseline_plots', exist_ok=True)
    output_path = "baseline_plots/combined_generalization_accuracy.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined plot: {output_path}")
    # plt.close()



def main():
    """Main function."""
    print("Loading taxonomy mappings...")
    generalization_maps = load_taxonomy_mappings()
    
    print("Loading results data...")
    results_data = load_results_data()
    
    print("Creating individual generalization mapping plots...")
    create_generalization_accuracy_plots(generalization_maps, results_data)
    
    print("Creating combined plot...")
    create_combined_plot(generalization_maps, results_data)
    
    print("Printing summary statistics...")
    print_summary_stats(generalization_maps, results_data)
    
    print("\nAll plots saved to 'baseline_plots/' directory")

if __name__ == "__main__":
    main() 