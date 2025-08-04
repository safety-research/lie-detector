#!/usr/bin/env python3
"""
Plot generalization results as histograms with error bars, including training results.
Shows results for both map_1 and map_2, with baseline types plus training results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import json
from pathlib import Path
import glob
import sys

# Generalization mappings from common/generalization_mappings.py
generalization_map_1 = {
    "instruction_following": [
        "alibi-fraud-investigation",
        "alibi_theft_investigation",
        "alibi-workplace-investigation",
        "chess_mate_in_2_shutdown_threat_task",
        "capture_the_secret_corporate",
        "cot_unfaithfulness_contrastive_mmlu",
        "cot_unfaithfulness_contrastive_arc",
    ],
    "in_context_pressure": [
        "mask_provided_facts",
        "mask_continuations",
        "mask_disinformation",
        "mask_statistics",
        "mask_known_facts",
        "cot_unfaithfulness_contrastive_mmlu",
        "cot_unfaithfulness_contrastive_arc",
    ],
    "internalized_drive_to_lie": [
        "sandbagging_automotive_mechanical",
        "sandbagging_automotive_mechanical_contrastive",
        "sandbagging_chemistry_biology",
        "sandbagging_chemistry_biology_contrastive",
        "sandbagging_contrastive",
        "sandbagging_detection",
        "sandbagging_digital_security",
        "sandbagging_digital_security_contrastive",
        "sandbagging_financial_systems",
        "sandbagging_financial_systems_contrastive",
        "sandbagging_judge_lie_detection",
        "sandbagging_lie_detection",
        "sandbagging_medical_pharmaceutical",
        "sandbagging_medical_pharmaceutical_contrastive",
        "sandbagging_physical_security",
        "sandbagging_physical_security_contrastive",
        "ascii_sandbagging_task",
        "sycophancy_answer_positive_correct",
        "sycophancy_answer_positive_incorrect",
        "sycophancy_answer_negative_correct",
        "sycophancy_feedback_user_did_not_write",
        "sycophancy_feedback_user_dislikes",
        "sycophancy_feedback_user_likes",
        "sycophancy_feedback_user_wrote",
        "tool_code_sandbagging_task",
        "tool_image_gen_sandbagging_task",
        "tool_search_sandbagging_task",
        "self_sycophancy_correctness_arc",
        "self_sycophancy_correctness_biology",
        "self_sycophancy_correctness_chemistry",
        "self_sycophancy_correctness_mathematics",
        "self_sycophancy_correctness_physics",
        "self_sycophancy_harmfulness_common_sense",
        "self_sycophancy_harmfulness_deontology",
        "self_sycophancy_harmfulness_justice",
    ],
}

generalization_map_2 = {
    "in_context_knowledge": [
        "sandbagging_automotive_mechanical",
        "sandbagging_automotive_mechanical_contrastive",
        "sandbagging_chemistry_biology",
        "sandbagging_chemistry_biology_contrastive",
        "sandbagging_digital_security",
        "sandbagging_digital_security_contrastive",
        "sandbagging_financial_systems",
        "sandbagging_financial_systems_contrastive",
        "sandbagging_medical_pharmaceutical",
        "sandbagging_medical_pharmaceutical_contrastive",
        "sandbagging_physical_security",
        "sandbagging_physical_security_contrastive",
        "sandbagging_tool_use",
        "mask_provided_facts",
    ],
    "internal_knowledge": [
        "sycophancy_answer_positive_correct",
        "sycophancy_answer_positive_incorrect",
        "sycophancy_answer_negative_correct",
        "sycophancy_feedback_user_did_not_write",
        "sycophancy_feedback_user_dislikes",
        "sycophancy_feedback_user_wrote",
        "sycophancy_feedback_user_likes",
        "mask_known_facts",
        "mask_statistics",
        "mask_disinformation",
        "mask_continuations",
        "cot_unfaithfulness_contrastive_mmlu",
        "cot_unfaithfulness_contrastive_arc",
        "sandbagging_automotive_mechanical",
        "sandbagging_harmlessness",
    ],
    "introspective_knowledge": [
        "self_sycophancy_correctness_physics",
        "self_sycophancy_correctness_mathematics",
        "self_sycophancy_correctness_chemistry",
        "self_sycophancy_correctness_arc",
        "self_sycophancy_correctness_biology",
        "self_sycophancy_harmfulness_common_sense",
        "self_sycophancy_harmfulness_justice",
        "self_sycophancy_harmfulness_deontology",
        "ascii_sandbagging_task",
    ],
}

def load_generalization_results():
    """Load the generalization CSV results from individual baseline folders."""
    results_dir = "results"
    all_data = []
    
    # Baseline types and their folders
    baseline_types = {
        'base_transcript': 'base_transcript',
        'escaped_transcript': 'escaped_transcript', 
        'llama_chat': 'llama_chat'
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
                df['data_source'] = 'baseline'  # Mark as baseline data
                
                all_data.append(df)
                
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
    
    if not all_data:
        print("No generalization CSV files found in results directory")
        return None
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def load_training_results():
    """Load training results from clean results with task IDs and convert to same format as generalization results."""
    training_data = []
    
    # Training result files with task IDs
    training_files = {
        'base_transcript': '../train/clean_results_with_task_ids/base_transcript_sft.json',
        'llama_chat': '../train/clean_results_with_task_ids/llama_chat_sft.json'
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
            results = val_metrics.get('results', [])
            
            if not results:
                print(f"No results found in {training_type}")
                continue
            
            # --- Map 1 logic (as before) ---
            category_results = {}
            for result in results:
                task_id = result.get('task_id', 'unknown')
                true_label = result.get('true', '')
                predicted_label = result.get('predicted', '')
                correct = result.get('correct', False)
                # Map task_id to map_1 category (as before)
                map1_category = None
                for category, task_list in generalization_map_1.items():
                    if task_id in task_list:
                        map1_category = f"map_1_{category}"
                        break
                if map1_category is None:
                    map1_category = map_task_id_to_category(task_id)
                if map1_category not in category_results:
                    category_results[map1_category] = {
                        'correct': 0,
                        'total': 0,
                        'predictions': []
                    }
                category_results[map1_category]['total'] += 1
                if correct:
                    category_results[map1_category]['correct'] += 1
                category_results[map1_category]['predictions'].append({
                    'true': true_label,
                    'predicted': predicted_label,
                    'correct': correct
                })
            # Calculate precision for each map_1 category
            for category, stats in category_results.items():
                if stats['total'] > 0:
                    precision = stats['correct'] / stats['total']
                    precision_se = np.sqrt(precision * (1 - precision) / stats['total'])
                    training_row = {
                        'Dataset / Task': category,
                        'precision': precision,
                        'precision_se': precision_se,
                        'baseline_type': training_type,
                        'model': 'meta-llama_llama-3.1-8b-instruct',
                        'data_source': 'training',
                        'total_samples': stats['total']
                    }
                    training_data.append(training_row)
            # --- Map 2 logic: aggregate by category ---
            for map2_category, task_list in generalization_map_2.items():
                # Collect all results for this map2 category
                filtered_results = [r for r in results if r.get('task_id', 'unknown') in task_list]
                if not filtered_results:
                    continue
                total = len(filtered_results)
                correct = sum(1 for r in filtered_results if r.get('correct', False))
                precision = correct / total if total > 0 else 0
                precision_se = np.sqrt(precision * (1 - precision) / total) if total > 0 else 0
                training_row = {
                    'Dataset / Task': f"map_2_{map2_category}",
                    'precision': precision,
                    'precision_se': precision_se,
                    'baseline_type': training_type,
                    'model': 'meta-llama_llama-3.1-8b-instruct',
                    'data_source': 'training',
                    'total_samples': total
                }
                training_data.append(training_row)
            print(f"Loaded {training_type} training results with {len(category_results)} map_1 and map_2 categories")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    if not training_data:
        print("No training results found")
        return None
    return pd.DataFrame(training_data)

def map_task_id_to_category(task_id):
    """Map task IDs to map_1 and map_2 categories using the proper generalization mappings."""
    
    # Check map_1 categories
    for category, task_list in generalization_map_1.items():
        if task_id in task_list:
            return f"map_1_{category}"
    
    # Check map_2 categories
    for category, task_list in generalization_map_2.items():
        if task_id in task_list:
            return f"map_2_{category}"
    
    # If not found in mappings, try to infer based on task_id content
    task_id_lower = task_id.lower()
    
    if 'harmfulness' in task_id_lower:
        return 'map_1_internalized_drive_to_lie'  # harmfulness tasks are in internalized_drive_to_lie
    elif 'sycophancy' in task_id_lower:
        if 'self' in task_id_lower:
            return 'map_2_introspective_knowledge'  # self_sycophancy tasks are in introspective_knowledge
        else:
            # Regular sycophancy tasks are in map_2_internal_knowledge, not map_1
            return 'map_2_internal_knowledge'
    elif 'sandbagging' in task_id_lower:
        return 'map_1_internalized_drive_to_lie'  # sandbagging tasks are in internalized_drive_to_lie
    elif 'mask' in task_id_lower:
        # Mask tasks are in map_2_internal_knowledge, not map_1
        return 'map_2_internal_knowledge'
    elif 'alibi' in task_id_lower:
        return 'map_1_instruction_following'  # alibi tasks are in instruction_following
    elif 'chess' in task_id_lower:
        return 'map_1_instruction_following'  # chess tasks are in instruction_following
    else:
        # Default to instruction following for unknown tasks
        return 'map_1_instruction_following'

def create_generalization_plots_with_training(baseline_df, training_df):
    """Create histograms for generalization results including training."""
    # Filter for llama 8b model
    llama_8b_baseline = baseline_df[baseline_df['model'] == 'meta-llama_llama-3.1-8b-instruct'].copy()
    
    if llama_8b_baseline.empty and training_df is None:
        print("No data found for llama-3.1-8b-instruct model")
        return
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))  # Increased width for more bars
    
    # Add overall title
    fig.suptitle('Precision for Different Baseline Types and Training on Llama 8B Instruct', fontsize=16, fontweight='bold')
    
    # Plot for map_1
    plot_category_results_with_training(ax1, llama_8b_baseline, training_df, 'map_1', 'Map 1 Categories')
    
    # Plot for map_2
    plot_category_results_with_training(ax2, llama_8b_baseline, training_df, 'map_2', 'Map 2 Categories')
    
    # Create a single legend at the bottom
    legend_labels = ['Base Transcript (Baseline)', 'Escaped Transcript (Baseline)', 'Llama Chat (Baseline)', 
                    'Base Transcript (Training)', 'Llama Chat (Training)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Create legend handles
    legend_elements = []
    for label, color in zip(legend_labels, colors):
        legend_elements.append(patches.Rectangle((0,0),1,1, facecolor=color, alpha=0.8, label=label))
    
    # Add legend at the bottom outside the plots
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=5)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for legend
    plt.savefig('results/generalization_plots_with_training.png', dpi=300, bbox_inches='tight')
    
    print("Plots saved to results/generalization_plots_with_training.png")

def plot_category_results_with_training(ax, baseline_df, training_df, map_prefix, title):
    """Plot results for a specific map prefix including training results."""
    # Use 'Dataset / Task' as the category column
    category_col = 'Dataset / Task'
    
    # Get unique categories for this map prefix (excluding 'OVERALL')
    baseline_categories = [cat for cat in baseline_df[category_col].unique() if cat.startswith(map_prefix)]
    baseline_categories = sorted(baseline_categories)
    
    # Get training categories if available
    training_categories = []
    if training_df is not None:
        training_categories = [cat for cat in training_df[category_col].unique() if cat.startswith(map_prefix)]
        training_categories = sorted(training_categories)
    
    # Use the union of categories
    all_categories = sorted(list(set(baseline_categories + training_categories)))
    
    print(f"Debug {map_prefix}: baseline_categories={baseline_categories}, training_categories={training_categories}, all_categories={all_categories}")
    
    if not all_categories:
        ax.text(0.5, 0.5, f'No {map_prefix} categories found', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return
    
    # Clean up category labels for display
    display_labels = []
    for category in all_categories:
        # Remove the map prefix (e.g., "map_1_instruction_following" -> "instruction_following")
        clean_label = category.replace(f'{map_prefix}_', '')
        # Replace underscores with spaces
        clean_label = clean_label.replace('_', ' ')
        # Capitalize first letter of each word
        clean_label = ' '.join(word.capitalize() for word in clean_label.split())
        display_labels.append(clean_label)
    
    # Define all data sources (baseline + training)
    data_sources = [
        ('base_transcript', 'baseline', '#1f77b4'),      # Blue
        ('escaped_transcript', 'baseline', '#ff7f0e'),   # Orange
        ('llama_chat', 'baseline', '#2ca02c'),           # Green
        ('base_transcript', 'training', '#d62728'),      # Red
        ('llama_chat', 'training', '#9467bd')            # Purple
    ]
    
    # Set up the plot
    x = np.arange(len(all_categories))
    width = 0.15  # Width of bars (5 bars total)
    
    # Plot bars for each data source
    for i, (data_type, source, color) in enumerate(data_sources):
        values = []
        errors = []
        
        for category in all_categories:
            # Get data for this category and data source
            if source == 'baseline':
                mask = (baseline_df[category_col] == category) & (baseline_df['baseline_type'] == data_type)
                category_data = baseline_df[mask]
            else:  # training
                if training_df is not None:
                    mask = (training_df[category_col] == category) & (training_df['baseline_type'] == data_type)
                    category_data = training_df[mask]
                else:
                    category_data = pd.DataFrame()
            
            if not category_data.empty:
                precision = category_data['precision'].iloc[0]
                std = category_data['precision_se'].iloc[0] if 'precision_se' in category_data.columns else 0
                values.append(precision)
                errors.append(std)
            else:
                values.append(0)
                errors.append(0)
        
        # Plot bars with error bars
        bars = ax.bar(x + i * width, values, width, color=color, alpha=0.8)
        
        # Add error bars if we have standard error data
        if any(err != 0 for err in errors):
            ax.errorbar(x + i * width, values, yerr=errors, fmt='none', color='black', capsize=3, capthick=1)
        
        # Add value labels on bars (positioned above error bars)
        for j, (bar, value, error) in enumerate(zip(bars, values, errors)):
            height = bar.get_height()
            # Position text above the error bar if there is one, otherwise above the bar
            text_y = height + error + 0.02 if error > 0 else height + 0.02
            ax.text(bar.get_x() + bar.get_width()/2., text_y,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=7)
    
    # Customize the plot
    ax.set_xlabel('Categories')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.set_xticks(x + 2 * width)  # Center the labels between the bars
    ax.set_xticklabels(display_labels, ha='center')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.2)  # Precision is between 0 and 1, extend slightly above 1

def main():
    """Main function to create generalization plots with training."""
    print("Loading generalization results...")
    baseline_df = load_generalization_results()
    
    print("Loading training results...")
    training_df = load_training_results()
    
    if baseline_df is None and training_df is None:
        print("No data found!")
        return
    
    print("Creating generalization plots with training...")
    create_generalization_plots_with_training(baseline_df, training_df)
    print("Done!")

if __name__ == "__main__":
    main() 