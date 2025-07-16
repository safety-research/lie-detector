#!/usr/bin/env python3
"""
Calculate standard errors for generalization categories from individual task results.
This adds precision_se columns to the generalization CSV files.
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path

def load_generalization_mappings():
    """Load the generalization mappings."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from common.generalization_mappings import map_1, map_2
    return map_1, map_2

def calculate_se_for_category(tasks_in_category, sorted_df):
    """Calculate standard error of the mean precision for tasks in a category."""
    task_se_values = []
    
    for task in tasks_in_category:
        # Find the task in the sorted dataframe
        task_data = sorted_df[sorted_df['Dataset / Task'] == task]
        if not task_data.empty:
            precision = task_data['precision'].iloc[0]
            total_samples = task_data['total_samples'].iloc[0]
            
            # Calculate standard error for this task using the formula:
            # SE = sqrt(precision * (1 - precision) / n)
            if total_samples > 0:
                se_task = np.sqrt(precision * (1 - precision) / total_samples)
                task_se_values.append(se_task)
    
    if len(task_se_values) > 1:
        # Calculate standard error of the mean across tasks
        # SE_avg = sqrt(sum(SE_i^2) / n_tasks)
        se_avg = np.sqrt(np.sum(np.array(task_se_values) ** 2) / len(task_se_values))
        return se_avg
    elif len(task_se_values) == 1:
        return task_se_values[0]
    else:
        return 0.0

def update_generalization_csv_with_se():
    """Update generalization CSV files with standard error data."""
    # Load generalization mappings
    map_1, map_2 = load_generalization_mappings()
    
    # Baseline types and their folders
    baseline_types = {
        'base_transcript': 'base_transcript',
        'escaped_transcript': 'escaped_transcript', 
        'llama_chat': 'llama_chat'
    }
    
    for baseline_type, folder in baseline_types.items():
        folder_path = os.path.join("results", folder)
        if not os.path.exists(folder_path):
            continue
            
        # Find sorted metrics CSV files for this baseline type
        pattern = os.path.join(folder_path, "sorted_metrics_*.csv")
        sorted_files = glob.glob(pattern)
        
        for sorted_file in sorted_files:
            try:
                # Load sorted metrics data
                sorted_df = pd.read_csv(sorted_file)
                
                # Extract model name from filename
                filename = os.path.basename(sorted_file)
                if 'llama_3.1_8b' in filename:
                    model_file = 'meta_llama_llama_3.1_8b_instruct'
                elif 'llama_3.3_70b' in filename:
                    model_file = 'meta_llama_llama_3.3_70b_instruct'
                else:
                    continue
                
                # Map baseline_type to file suffix
                if baseline_type == 'base_transcript':
                    file_suffix = 'base'
                elif baseline_type == 'escaped_transcript':
                    file_suffix = 'escaped'
                elif baseline_type == 'llama_chat':
                    file_suffix = 'llama'
                else:
                    file_suffix = baseline_type
                
                # Find corresponding generalization CSV
                gen_file = os.path.join(folder_path, f"generalization_metrics_{model_file}_{file_suffix}.csv")
                if not os.path.exists(gen_file):
                    print(f"No generalization CSV found: {gen_file}")
                    continue
                
                gen_df = pd.read_csv(gen_file)
                
                # Calculate standard errors for each category
                se_values = []
                
                for _, row in gen_df.iterrows():
                    category = row['Dataset / Task']
                    
                    if category == 'OVERALL':
                        # For overall, calculate SE across all tasks
                        all_precisions = sorted_df['precision'].tolist()
                        all_samples = sorted_df['total_samples'].tolist()
                        
                        # Calculate SE for each task, then SE of the mean
                        task_se_values = []
                        for precision, samples in zip(all_precisions, all_samples):
                            if samples > 0:
                                se_task = np.sqrt(precision * (1 - precision) / samples)
                                task_se_values.append(se_task)
                        
                        if len(task_se_values) > 1:
                            se_avg = np.sqrt(np.sum(np.array(task_se_values) ** 2) / len(task_se_values))
                            se_values.append(se_avg)
                        elif len(task_se_values) == 1:
                            se_values.append(task_se_values[0])
                        else:
                            se_values.append(0.0)
                            
                    elif category.startswith('map_1_'):
                        # Find tasks in map_1 category
                        category_name = category.replace('map_1_', '')
                        if category_name in map_1:
                            tasks_in_category = map_1[category_name]
                            se_val = calculate_se_for_category(tasks_in_category, sorted_df)
                            se_values.append(se_val)
                        else:
                            se_values.append(0.0)
                    elif category.startswith('map_2_'):
                        # Find tasks in map_2 category
                        category_name = category.replace('map_2_', '')
                        if category_name in map_2:
                            tasks_in_category = map_2[category_name]
                            se_val = calculate_se_for_category(tasks_in_category, sorted_df)
                            se_values.append(se_val)
                        else:
                            se_values.append(0.0)
                    else:
                        se_values.append(0.0)
                
                # Add standard error column (replace existing if present)
                gen_df['precision_se'] = se_values
                
                # Save updated generalization CSV
                gen_df.to_csv(gen_file, index=False)
                print(f"Updated {gen_file} with standard errors")
                
            except Exception as e:
                print(f"Error processing {sorted_file}: {e}")

def main():
    """Main function to calculate and add standard errors."""
    print("Calculating standard errors for generalization categories...")
    update_generalization_csv_with_se()
    print("Done!")

if __name__ == "__main__":
    import sys
    main() 