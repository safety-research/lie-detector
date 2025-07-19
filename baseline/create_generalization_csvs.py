#!/usr/bin/env python3
"""
Create generalization CSV files for baseline evaluation results.

This script processes all baseline evaluation results and creates CSV files
with aggregated metrics for each generalization category defined in generalization_mappings.py.
Creates 7 rows: 1 overall + 6 generalization categories (3 from map_1 + 3 from map_2).
"""

import re
import json
import os
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import sys

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
            if part in ['base', 'escaped', 'llama']:
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
        print(f"Error loading {filepath}: {e}")
        return []

def parse_json_results_with_subtasks(filepath: str) -> List[Dict[str, Any]]:
    """Parse JSON results file and extract both overall and subtask metrics (for backward compatibility)."""
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
            if part in ['base', 'escaped', 'llama']:
                baseline_type = part
                break
        
        if not baseline_type:
            baseline_type = "unknown"
        
        results = []
        
        # Extract overall metrics
        metrics = data.get('metrics', {})
        overall_row = {
            'Dataset / Task': 'OVERALL',
            'total_samples': metrics.get('parseable_samples', 0) + metrics.get('filtered_samples', 0),
            'parseable_samples': metrics.get('parseable_samples', 0),
            'parse_rate': metrics.get('parseable_samples', 0) / max(1, metrics.get('parseable_samples', 0) + metrics.get('filtered_samples', 0)),
            'accuracy': metrics.get('accuracy', 0.0),
            'precision': metrics.get('precision', 0.0),
            'recall': metrics.get('recall', 0.0),
            'f1_score': metrics.get('f1_score', 0.0),
            'TP': metrics.get('true_positives', 0),
            'TN': metrics.get('true_negatives', 0),
            'FP': metrics.get('false_positives', 0),
            'FN': metrics.get('false_negatives', 0),
            'support': metrics.get('parseable_samples', 0) + metrics.get('filtered_samples', 0),
            'model': model_name,
            'baseline_type': baseline_type
        }
        results.append(overall_row)
        
        # Extract subtask metrics
        subtask_results = data.get('subtask_results', {})
        for subtask_name, subtask_data in subtask_results.items():
            samples = subtask_data.get('samples', [])
            
            # Calculate metrics for this subtask
            total_samples = len(samples)
            parseable_samples = sum(1 for s in samples if s.get('is_parseable', False))
            correct_predictions = sum(1 for s in samples if s.get('correct') is True)
            
            # Calculate confusion matrix
            tp = tn = fp = fn = 0
            for sample in samples:
                if not sample.get('is_parseable', False):
                    continue
                target = sample.get('target', '')
                prediction = sample.get('extracted_prediction', '')
                correct = sample.get('correct', False)
                
                if target == 'A' and prediction == 'A':
                    tp += 1
                elif target == 'B' and prediction == 'B':
                    tn += 1
                elif target == 'B' and prediction == 'A':
                    fp += 1
                elif target == 'A' and prediction == 'B':
                    fn += 1
            
            # Calculate derived metrics
            accuracy = correct_predictions / max(parseable_samples, 1) if parseable_samples > 0 else 0
            precision = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0
            recall = tp / max(tp + fn, 1) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / max(precision + recall, 1e-8) if (precision + recall) > 0 else 0
            parse_rate = parseable_samples / max(total_samples, 1)
            
            subtask_row = {
                'Dataset / Task': subtask_name,
                'total_samples': total_samples,
                'parseable_samples': parseable_samples,
                'parse_rate': parse_rate,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'TP': tp,
                'TN': tn,
                'FP': fp,
                'FN': fn,
                'support': total_samples,
                'model': model_name,
                'baseline_type': baseline_type
            }
            results.append(subtask_row)
        
        return results
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return []

def find_results_files() -> Dict[str, List[str]]:
    """Find all results files organized by baseline type."""
    results_dir = Path("results")
    if not results_dir.exists():
        print(f"Results directory {results_dir} does not exist!")
        return {}
    
    baseline_files = {}
    
    # Find all baseline type directories
    for baseline_dir in results_dir.iterdir():
        if baseline_dir.is_dir():
            baseline_type = baseline_dir.name
            # Look for both .txt and .json files
            txt_files = list(baseline_dir.glob("*.txt"))
            json_files = list(baseline_dir.glob("*.json"))
            all_files = txt_files + json_files
            if all_files:
                baseline_files[baseline_type] = [str(f) for f in all_files]
    
    return baseline_files

def aggregate_metrics_for_category(subtask_results: List[Dict[str, Any]], category_tasks: List[str]) -> Dict[str, Any] | None:
    """Aggregate metrics for a specific generalization category."""
    # Filter results to only include tasks in this category
    category_results = [r for r in subtask_results if r['Dataset / Task'] in category_tasks]
    
    if not category_results:
        return None
    
    # Aggregate metrics
    total_samples = sum(r['total_samples'] for r in category_results)
    parseable_samples = sum(r['parseable_samples'] for r in category_results)
    total_tp = sum(r['TP'] for r in category_results)
    total_tn = sum(r['TN'] for r in category_results)
    total_fp = sum(r['FP'] for r in category_results)
    total_fn = sum(r['FN'] for r in category_results)
    
    # Calculate aggregated metrics
    parse_rate = parseable_samples / max(total_samples, 1)
    accuracy = (total_tp + total_tn) / max(parseable_samples, 1) if parseable_samples > 0 else 0
    precision = total_tp / max(total_tp + total_fp, 1) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / max(total_tp + total_fn, 1) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / max(precision + recall, 1e-8) if (precision + recall) > 0 else 0
    
    return {
        'total_samples': total_samples,
        'parseable_samples': parseable_samples,
        'parse_rate': parse_rate,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'TP': total_tp,
        'TN': total_tn,
        'FP': total_fp,
        'FN': total_fn,
        'support': total_samples
    }

def create_generalization_csvs_in_results(baseline_files: Dict[str, List[str]]):
    """Create generalization CSV files for each baseline-model combination."""
    print(f"\nCreating generalization CSV files in results directory...")
    
    # Process each baseline type
    for baseline_type, files in baseline_files.items():
        print(f"\nProcessing {baseline_type} baseline for generalization CSVs...")
        
        all_baseline_results = []
        for filepath in files:
            print(f"  Processing: {os.path.basename(filepath)}")
            if filepath.endswith('.txt'):
                results = parse_text_results_with_subtasks(filepath)
            else:
                results = parse_json_results_with_subtasks(filepath)
            if results:
                all_baseline_results.extend(results)
        
        if all_baseline_results:
            # Group by model
            model_groups = {}
            for result in all_baseline_results:
                model = result['model']
                if model not in model_groups:
                    model_groups[model] = []
                model_groups[model].append(result)
            
            # Create generalization CSV for each model
            for model, results in model_groups.items():
                print(f"    Creating generalization CSV for {model}...")
                
                # Separate overall and subtask results
                overall_result = None
                subtask_results = []
                
                for result in results:
                    if result['Dataset / Task'] == 'OVERALL':
                        overall_result = result
                    else:
                        subtask_results.append(result)
                
                if not overall_result:
                    print(f"      Warning: No overall results found for {model}")
                    continue
                
                # Create generalization rows
                generalization_rows = []
                
                # Add overall row
                generalization_rows.append(overall_result)
                
                # Add map_1 categories
                for category_name, category_tasks in map_1.items():
                    aggregated_metrics = aggregate_metrics_for_category(subtask_results, category_tasks)
                    if aggregated_metrics:
                        category_row = {
                            'Dataset / Task': f"map_1_{category_name}",
                            'model': model,
                            'baseline_type': baseline_type,
                            **aggregated_metrics
                        }
                        generalization_rows.append(category_row)
                        print(f"      Added map_1_{category_name}: {len([r for r in subtask_results if r['Dataset / Task'] in category_tasks])} tasks")
                
                # Add map_2 categories
                for category_name, category_tasks in map_2.items():
                    aggregated_metrics = aggregate_metrics_for_category(subtask_results, category_tasks)
                    if aggregated_metrics:
                        category_row = {
                            'Dataset / Task': f"map_2_{category_name}",
                            'model': model,
                            'baseline_type': baseline_type,
                            **aggregated_metrics
                        }
                        generalization_rows.append(category_row)
                        print(f"      Added map_2_{category_name}: {len([r for r in subtask_results if r['Dataset / Task'] in category_tasks])} tasks")
                
                # Create dataframe
                df = pd.DataFrame(generalization_rows)
                
                # Create sorting columns for proper ordering
                df['is_overall'] = df['Dataset / Task'] == 'OVERALL'
                df['mapping_group'] = df['Dataset / Task'].apply(lambda x: 
                    'overall' if x == 'OVERALL' else 
                    'map_1' if x.startswith('map_1_') else 
                    'map_2' if x.startswith('map_2_') else 'other'
                )
                
                # Sort: OVERALL first, then map_1 categories by precision, then map_2 categories by precision
                df = df.sort_values(['is_overall', 'mapping_group', 'precision'], 
                                  ascending=[False, True, False])
                df = df.drop(['is_overall', 'mapping_group'], axis=1)
                
                # Create filename
                model_safe = model.replace('/', '_').replace('-', '_')
                csv_filename = f"generalization_metrics_{model_safe}.csv"
                
                # Place in the respective baseline folder
                baseline_folder = f"results/{baseline_type}"
                csv_path = os.path.join(baseline_folder, csv_filename)
                
                # Save to CSV
                df.to_csv(csv_path, index=False)
                print(f"      Created generalization CSV: {csv_path}")
                print(f"        - {len(df)} rows (1 overall + 6 generalization categories)")
                print(f"        - Precision range: {df['precision'].max():.4f} to {df['precision'].min():.4f}")
    
    print(f"\nGeneralization CSV files created in respective baseline folders within results/")

def main():
    """Main function to create generalization CSVs."""
    print("Creating generalization CSV files for baseline evaluation results...")
    
    # Find all results files
    baseline_files = find_results_files()
    
    if not baseline_files:
        print("No results files found!")
        return
    
    # Create generalization CSV files in results directory
    create_generalization_csvs_in_results(baseline_files)
    
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE!")
    print(f"{'='*60}")
    print("Files created:")
    print("Generalization CSVs in: results/[baseline_type]/")
    print("Each CSV contains 7 rows:")
    print("  1. OVERALL")
    print("  2. map_1_instruction_following")
    print("  3. map_1_in_context_pressure")
    print("  4. map_1_internalized_drive_to_lie")
    print("  5. map_2_in_context_knowledge")
    print("  6. map_2_internal_knowledge")
    print("  7. map_2_introspective_knowledge")

if __name__ == "__main__":
    main() 