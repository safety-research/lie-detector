#!/usr/bin/env python3
"""
Complete analysis pipeline for baseline evaluation results.

This script combines three main functions:
1. generate_generalization_csvs() - Creates generalization CSV files
2. calculate_standard_errors() - Adds standard error calculations
3. create_plots() - Generates visualization plots

Usage:
    python generate_complete_analysis.py /path/to/results/directory
"""

import re
import json
import os
import glob
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Add the common directory to the path to import generalization_mappings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.generalization_mappings import generalization_map_1 as map_1, generalization_map_2 as map_2

def parse_text_results_with_subtasks(filepath: str) -> List[Dict[str, Any]]:
    """Parse text results file and extract both overall and subtask metrics."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Extract model name from filename
        filename = os.path.basename(filepath)
        parts = filename.replace('.txt', '').split('_')
        
        # Find model name in the parts
        model_name = None
        for i, part in enumerate(parts):
            if 'meta-llama' in part or 'llama' in part:
                model_name = '_'.join(parts[i:i+3])  # Take 3 parts for full model name
                break
        
        if not model_name:
            model_name = "unknown"
        
        # Extract baseline type from filename
        baseline_type = None
        for part in parts:
            if part in ['base', 'escaped', 'llama', 'rowans']:
                baseline_type = part
                break
        
        if not baseline_type:
            baseline_type = "unknown"
        
        results = []
        
        # Extract overall metrics
        overall_metrics = {}
        
        # Extract overall metrics
        accuracy_match = re.search(r'Accuracy:\s*([\d.]+)', content)
        if accuracy_match:
            overall_metrics['accuracy'] = float(accuracy_match.group(1))
        
        precision_match = re.search(r'Precision:\s*([\d.]+)', content)
        if precision_match:
            overall_metrics['precision'] = float(precision_match.group(1))
        
        recall_match = re.search(r'Recall:\s*([\d.]+)', content)
        if recall_match:
            overall_metrics['recall'] = float(recall_match.group(1))
        
        f1_match = re.search(r'F1 Score:\s*([\d.]+)', content)
        if f1_match:
            overall_metrics['f1_score'] = float(f1_match.group(1))
        
        # Extract overall sample counts
        total_samples_match = re.search(r'Total Samples:\s*(\d+)', content)
        if total_samples_match:
            overall_metrics['total_samples'] = int(total_samples_match.group(1))
        
        parseable_match = re.search(r'Parseable samples:\s*(\d+)/(\d+)', content)
        if parseable_match:
            overall_metrics['parseable_samples'] = int(parseable_match.group(1))
            overall_metrics['total_samples'] = int(parseable_match.group(2))
        
        # Extract overall confusion matrix
        tp_match = re.search(r'True Positives:\s*(\d+)', content)
        if tp_match:
            overall_metrics['TP'] = int(tp_match.group(1))
        
        tn_match = re.search(r'True Negatives:\s*(\d+)', content)
        if tn_match:
            overall_metrics['TN'] = int(tn_match.group(1))
        
        fp_match = re.search(r'False Positives:\s*(\d+)', content)
        if fp_match:
            overall_metrics['FP'] = int(fp_match.group(1))
        
        fn_match = re.search(r'False Negatives:\s*(\d+)', content)
        if fn_match:
            overall_metrics['FN'] = int(fn_match.group(1))
        
        # Calculate overall parse rate
        if overall_metrics.get('total_samples', 0) > 0:
            overall_metrics['parse_rate'] = overall_metrics.get('parseable_samples', 0) / overall_metrics.get('total_samples', 1)
        else:
            overall_metrics['parse_rate'] = 0.0
        
        # Add overall row
        overall_row = {
            'Dataset / Task': 'OVERALL',
            'total_samples': overall_metrics.get('total_samples', 0),
            'parseable_samples': overall_metrics.get('parseable_samples', 0),
            'parse_rate': overall_metrics.get('parse_rate', 0.0),
            'accuracy': overall_metrics.get('accuracy', 0.0),
            'precision': overall_metrics.get('precision', 0.0),
            'recall': overall_metrics.get('recall', 0.0),
            'f1_score': overall_metrics.get('f1_score', 0.0),
            'TP': overall_metrics.get('TP', 0),
            'TN': overall_metrics.get('TN', 0),
            'FP': overall_metrics.get('FP', 0),
            'FN': overall_metrics.get('FN', 0),
            'support': overall_metrics.get('total_samples', 0),
            'model': model_name,
            'baseline_type': baseline_type
        }
        results.append(overall_row)
        
        # Extract subtask metrics
        subtask_sections = re.findall(r'SUBTASK:\s*([^\n]+)\n-+\n(.*?)(?=SUBTASK:|$)', content, re.DOTALL)
        
        for subtask_name, subtask_content in subtask_sections:
            subtask_name = subtask_name.strip()
            
            # Extract subtask metrics
            subtask_metrics = {}
            
            # Sample counts
            total_match = re.search(r'Total samples:\s*(\d+)', subtask_content)
            if total_match:
                subtask_metrics['total_samples'] = int(total_match.group(1))
            
            parseable_match = re.search(r'Parseable samples:\s*(\d+)', subtask_content)
            if parseable_match:
                subtask_metrics['parseable_samples'] = int(parseable_match.group(1))
            
            # Performance metrics
            accuracy_match = re.search(r'Accuracy:\s*([\d.]+)', subtask_content)
            if accuracy_match:
                subtask_metrics['accuracy'] = float(accuracy_match.group(1))
            
            precision_match = re.search(r'Precision:\s*([\d.]+)', subtask_content)
            if precision_match:
                subtask_metrics['precision'] = float(precision_match.group(1))
            
            recall_match = re.search(r'Recall:\s*([\d.]+)', subtask_content)
            if recall_match:
                subtask_metrics['recall'] = float(recall_match.group(1))
            
            f1_match = re.search(r'F1 Score:\s*([\d.]+)', subtask_content)
            if f1_match:
                subtask_metrics['f1_score'] = float(f1_match.group(1))
            
            # Confusion matrix
            tp_match = re.search(r'TP:\s*(\d+)', subtask_content)
            if tp_match:
                subtask_metrics['TP'] = int(tp_match.group(1))
            
            tn_match = re.search(r'TN:\s*(\d+)', subtask_content)
            if tn_match:
                subtask_metrics['TN'] = int(tn_match.group(1))
            
            fp_match = re.search(r'FP:\s*(\d+)', subtask_content)
            if fp_match:
                subtask_metrics['FP'] = int(fp_match.group(1))
            
            fn_match = re.search(r'FN:\s*(\d+)', subtask_content)
            if fn_match:
                subtask_metrics['FN'] = int(fn_match.group(1))
            
            # Calculate parse rate
            if subtask_metrics.get('total_samples', 0) > 0:
                subtask_metrics['parse_rate'] = subtask_metrics.get('parseable_samples', 0) / subtask_metrics.get('total_samples', 1)
            else:
                subtask_metrics['parse_rate'] = 0.0
            
            # Add subtask row
            subtask_row = {
                'Dataset / Task': subtask_name,
                'total_samples': subtask_metrics.get('total_samples', 0),
                'parseable_samples': subtask_metrics.get('parseable_samples', 0),
                'parse_rate': subtask_metrics.get('parse_rate', 0.0),
                'accuracy': subtask_metrics.get('accuracy', 0.0),
                'precision': subtask_metrics.get('precision', 0.0),
                'recall': subtask_metrics.get('recall', 0.0),
                'f1_score': subtask_metrics.get('f1_score', 0.0),
                'TP': subtask_metrics.get('TP', 0),
                'TN': subtask_metrics.get('TN', 0),
                'FP': subtask_metrics.get('FP', 0),
                'FN': subtask_metrics.get('FN', 0),
                'support': subtask_metrics.get('total_samples', 0),
                'model': model_name,
                'baseline_type': baseline_type
            }
            results.append(subtask_row)
        
        return results
        
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return []

def parse_json_results_with_subtasks(filepath: str) -> List[Dict[str, Any]]:
    """Parse JSON results file and extract both overall and subtask metrics."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Extract model name from filename
        filename = os.path.basename(filepath)
        parts = filename.replace('.json', '').split('_')
        
        # Find model name in the parts
        model_name = None
        for i, part in enumerate(parts):
            if 'meta-llama' in part or 'llama' in part:
                model_name = '_'.join(parts[i:i+3])  # Take 3 parts for full model name
                break
        
        if not model_name:
            model_name = "unknown"
        
        # Extract baseline type from filename
        baseline_type = None
        for part in parts:
            if part in ['base', 'escaped', 'llama', 'rowans']:
                baseline_type = part
                break
        
        if not baseline_type:
            baseline_type = "unknown"
        
        results = []
        
        # Extract overall metrics
        overall_metrics = data.get('overall', {})
        
        # Add overall row
        overall_row = {
            'Dataset / Task': 'OVERALL',
            'total_samples': overall_metrics.get('total_samples', 0),
            'parseable_samples': overall_metrics.get('parseable_samples', 0),
            'parse_rate': overall_metrics.get('parse_rate', 0.0),
            'accuracy': overall_metrics.get('accuracy', 0.0),
            'precision': overall_metrics.get('precision', 0.0),
            'recall': overall_metrics.get('recall', 0.0),
            'f1_score': overall_metrics.get('f1_score', 0.0),
            'TP': overall_metrics.get('TP', 0),
            'TN': overall_metrics.get('TN', 0),
            'FP': overall_metrics.get('FP', 0),
            'FN': overall_metrics.get('FN', 0),
            'support': overall_metrics.get('total_samples', 0),
            'model': model_name,
            'baseline_type': baseline_type
        }
        results.append(overall_row)
        
        # Extract subtask metrics
        subtasks = data.get('subtasks', {})
        for subtask_name, subtask_metrics in subtasks.items():
            subtask_row = {
                'Dataset / Task': subtask_name,
                'total_samples': subtask_metrics.get('total_samples', 0),
                'parseable_samples': subtask_metrics.get('parseable_samples', 0),
                'parse_rate': subtask_metrics.get('parse_rate', 0.0),
                'accuracy': subtask_metrics.get('accuracy', 0.0),
                'precision': subtask_metrics.get('precision', 0.0),
                'recall': subtask_metrics.get('recall', 0.0),
                'f1_score': subtask_metrics.get('f1_score', 0.0),
                'TP': subtask_metrics.get('TP', 0),
                'TN': subtask_metrics.get('TN', 0),
                'FP': subtask_metrics.get('FP', 0),
                'FN': subtask_metrics.get('FN', 0),
                'support': subtask_metrics.get('total_samples', 0),
                'model': model_name,
                'baseline_type': baseline_type
            }
            results.append(subtask_row)
        
        return results
        
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return []

def find_results_files(results_dir: str) -> Dict[str, List[str]]:
    """Find all results files in the results directory."""
    baseline_files = {}
    
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
        
        # Find all .txt and .json files
        txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
        json_files = glob.glob(os.path.join(folder_path, "*.json"))
        
        baseline_files[baseline_type] = txt_files + json_files
    
    return baseline_files

def aggregate_metrics_for_category(subtask_results: List[Dict[str, Any]], category_tasks: List[str]) -> Dict[str, Any] | None:
    """Aggregate metrics for a specific category."""
    category_data = []
    
    for task in category_tasks:
        # Find the task in subtask results
        task_data = None
        for result in subtask_results:
            if result['Dataset / Task'] == task:
                task_data = result
                break
        
        if task_data:
            category_data.append(task_data)
    
    if not category_data:
        return None
    
    # Aggregate metrics
    total_samples = sum(item['total_samples'] for item in category_data)
    parseable_samples = sum(item['parseable_samples'] for item in category_data)
    
    # Calculate weighted average for precision, recall, f1
    weighted_precision = 0
    weighted_recall = 0
    weighted_f1 = 0
    total_weight = 0
    
    for item in category_data:
        weight = item['total_samples']
        if weight > 0:
            weighted_precision += item['precision'] * weight
            weighted_recall += item['recall'] * weight
            weighted_f1 += item['f1_score'] * weight
            total_weight += weight
    
    if total_weight > 0:
        weighted_precision /= total_weight
        weighted_recall /= total_weight
        weighted_f1 /= total_weight
    
    # Sum confusion matrix values
    total_tp = sum(item['TP'] for item in category_data)
    total_tn = sum(item['TN'] for item in category_data)
    total_fp = sum(item['FP'] for item in category_data)
    total_fn = sum(item['FN'] for item in category_data)
    
    # Calculate accuracy from confusion matrix
    accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn) if (total_tp + total_tn + total_fp + total_fn) > 0 else 0
    
    # Calculate parse rate
    parse_rate = parseable_samples / total_samples if total_samples > 0 else 0
    
    return {
        'total_samples': total_samples,
        'parseable_samples': parseable_samples,
        'parse_rate': parse_rate,
        'accuracy': accuracy,
        'precision': weighted_precision,
        'recall': weighted_recall,
        'f1_score': weighted_f1,
        'TP': total_tp,
        'TN': total_tn,
        'FP': total_fp,
        'FN': total_fn,
        'support': total_samples
    }

def generate_generalization_csvs(results_dir: str):
    """Generate generalization CSV files for all baseline results."""
    print("Finding results files...")
    baseline_files = find_results_files(results_dir)
    
    if not baseline_files:
        print("No results files found")
        return
    
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
        if baseline_type not in baseline_files:
            continue
        
        folder_path = os.path.join(results_dir, folder)
        files = baseline_files[baseline_type]
        
        print(f"\nProcessing {baseline_type}...")
        
        for filepath in files:
            print(f"  Processing {os.path.basename(filepath)}")
            
            # Parse results
            if filepath.endswith('.txt'):
                results = parse_text_results_with_subtasks(filepath)
            elif filepath.endswith('.json'):
                results = parse_json_results_with_subtasks(filepath)
            else:
                continue
            
            if not results:
                continue
            
            # Extract model name from first result
            model_name = results[0]['model']
            baseline_type_from_file = results[0]['baseline_type']
            
            # Create DataFrame
            df = pd.DataFrame(results)
            
            # Create generalization categories
            generalization_results = []
            
            # Add overall row
            overall_data = df[df['Dataset / Task'] == 'OVERALL'].iloc[0].to_dict()
            generalization_results.append(overall_data)
            
            # Process map_1 categories
            for category_name, tasks in map_1.items():
                category_data = aggregate_metrics_for_category(results, tasks)
                if category_data:
                    category_data['Dataset / Task'] = f'map_1_{category_name}'
                    category_data['model'] = model_name
                    category_data['baseline_type'] = baseline_type_from_file
                    generalization_results.append(category_data)
            
            # Process map_2 categories
            for category_name, tasks in map_2.items():
                category_data = aggregate_metrics_for_category(results, tasks)
                if category_data:
                    category_data['Dataset / Task'] = f'map_2_{category_name}'
                    category_data['model'] = model_name
                    category_data['baseline_type'] = baseline_type_from_file
                    generalization_results.append(category_data)
            
            # Create generalization DataFrame
            gen_df = pd.DataFrame(generalization_results)
            
            # Sort: OVERALL first, then map_1 categories by precision, then map_2 categories by precision
            overall_row = gen_df[gen_df['Dataset / Task'] == 'OVERALL']
            map_1_rows = gen_df[gen_df['Dataset / Task'].str.startswith('map_1_')].sort_values('precision', ascending=False)
            map_2_rows = gen_df[gen_df['Dataset / Task'].str.startswith('map_2_')].sort_values('precision', ascending=False)
            
            sorted_gen_df = pd.concat([overall_row, map_1_rows, map_2_rows], ignore_index=True)
            
            # Save generalization CSV
            output_filename = f"generalization_metrics_{model_name}_{baseline_type_from_file}.csv"
            output_path = os.path.join(folder_path, output_filename)
            sorted_gen_df.to_csv(output_path, index=False)
            print(f"    Saved {output_filename}")
            
            # Also save sorted metrics CSV for standard error calculation
            sorted_metrics_filename = f"sorted_metrics_{model_name}_{baseline_type_from_file}.csv"
            sorted_metrics_path = os.path.join(folder_path, sorted_metrics_filename)
            df.to_csv(sorted_metrics_path, index=False)
            print(f"    Saved {sorted_metrics_filename}")

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

def calculate_standard_errors(results_dir: str):
    """Calculate and add standard errors to generalization CSV files."""
    print("Calculating standard errors for generalization categories...")
    
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
                elif baseline_type == 'llama_chat_reasoning':
                    file_suffix = 'llama'
                elif baseline_type == 'base_transcript_reasoning':
                    file_suffix = 'base'
                elif baseline_type == 'rowans_escaped_transcript':
                    file_suffix = 'rowans'
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

def load_generalization_results(results_dir: str):
    """Load the generalization CSV results from individual baseline folders."""
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

def create_plots(results_dir: str):
    """Create histograms for generalization results."""
    print("Loading generalization results...")
    df = load_generalization_results(results_dir)
    
    if df is None:
        return
    
    print("Creating generalization plots...")
    
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
    
    # Save plot to results directory
    output_path = os.path.join(results_dir, 'generalization_plots.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plots saved to {output_path}")

def main():
    """Main function to run the complete analysis pipeline."""
    parser = argparse.ArgumentParser(description='Generate complete analysis for baseline evaluation results')
    parser.add_argument('results_dir', help='Path to the results directory')
    parser.add_argument('--skip-csvs', action='store_true', help='Skip CSV generation step')
    parser.add_argument('--skip-se', action='store_true', help='Skip standard error calculation step')
    parser.add_argument('--skip-plots', action='store_true', help='Skip plot generation step')
    
    args = parser.parse_args()
    
    # Validate results directory
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory '{args.results_dir}' does not exist")
        sys.exit(1)
    
    print(f"Running complete analysis on results directory: {args.results_dir}")
    
    # Step 1: Generate generalization CSVs
    if not args.skip_csvs:
        print("\n" + "="*50)
        print("STEP 1: Generating generalization CSV files")
        print("="*50)
        generate_generalization_csvs(args.results_dir)
    
    # Step 2: Calculate standard errors
    if not args.skip_se:
        print("\n" + "="*50)
        print("STEP 2: Calculating standard errors")
        print("="*50)
        calculate_standard_errors(args.results_dir)
    
    # Step 3: Create plots
    if not args.skip_plots:
        print("\n" + "="*50)
        print("STEP 3: Creating visualization plots")
        print("="*50)
        create_plots(args.results_dir)
    
    print("\n" + "="*50)
    print("Complete analysis finished!")
    print("="*50)

if __name__ == "__main__":
    main() 