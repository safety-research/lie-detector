#!/usr/bin/env python3
"""
Post-process baseline results to create balanced datasets.

For each subtask, find the number of did_lie=True and did_lie=False samples,
then sample the majority class down to match the minority class count
to create the largest possible balanced dataset for each subtask.
"""

import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict

# Configuration
RESULTS_DIR = "/Users/dipikakhullar/Desktop/lie-detector/baseline/organized_evaluation_20250721_215718/results"
BALANCED_RESULTS_DIR = "/Users/dipikakhullar/Desktop/lie-detector/baseline/organized_evaluation_20250721_215718/balanced_results"

# Baseline types
BASELINE_TYPES = ['base_transcript', 'llama_chat', 'llama_chat_reasoning', 'rowans_escaped_transcript']

def count_did_lie_samples(samples: List[Dict]) -> Tuple[int, int]:
    """Count the number of did_lie=True and did_lie=False samples."""
    true_count = sum(1 for sample in samples if sample.get('original_did_lie', False))
    false_count = sum(1 for sample in samples if not sample.get('original_did_lie', False))
    return true_count, false_count

def balance_samples(samples: List[Dict]) -> List[Dict]:
    """Balance samples by sampling the majority class down to match minority class count."""
    true_samples = [s for s in samples if s.get('original_did_lie', False)]
    false_samples = [s for s in samples if not s.get('original_did_lie', False)]
    
    true_count = len(true_samples)
    false_count = len(false_samples)
    
    print(f"    Original counts - True: {true_count}, False: {false_count}")
    
    if true_count == 0 or false_count == 0:
        print(f"    Warning: Cannot balance - one class has 0 samples")
        return samples
    
    # Determine which class is minority and majority
    if true_count <= false_count:
        minority_samples = true_samples
        majority_samples = false_samples
        minority_count = true_count
        majority_count = false_count
        minority_class = "True"
    else:
        minority_samples = false_samples
        majority_samples = true_samples
        minority_count = false_count
        majority_count = true_count
        minority_class = "False"
    
    # Sample majority class down to minority class count
    if majority_count > minority_count:
        sampled_majority = random.sample(majority_samples, minority_count)
        balanced_samples = minority_samples + sampled_majority
        print(f"    Balanced counts - {minority_class}: {minority_count}, Other: {minority_count}")
    else:
        balanced_samples = samples
        print(f"    Already balanced")
    
    return balanced_samples

def process_json_file(json_path: Path, baseline_type: str) -> Dict[str, Any]:
    """Process a single JSON file to create balanced results."""
    print(f"Processing: {json_path.name}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Create balanced version
    balanced_data = data.copy()
    
    # Process each subtask
    for subtask_name, subtask_data in data['subtask_results'].items():
        samples = subtask_data.get('samples', [])
        
        if not samples:
            print(f"  {subtask_name}: No samples found")
            continue
        
        print(f"  {subtask_name}:")
        
        # Balance the samples
        balanced_samples = balance_samples(samples)
        
        # Update the subtask data
        balanced_data['subtask_results'][subtask_name]['samples'] = balanced_samples
        
        # Recalculate metrics based on balanced samples
        true_count = sum(1 for s in balanced_samples if s.get('original_did_lie', False))
        false_count = sum(1 for s in balanced_samples if not s.get('original_did_lie', False))
        total_samples = len(balanced_samples)
        
        # Update metrics with balanced data
        metrics = balanced_data['subtask_results'][subtask_name]['metrics']
        metrics['total_samples'] = total_samples
        metrics['parseable_samples'] = total_samples  # All balanced samples are parseable
        metrics['filtered_samples'] = 0  # No filtering in balanced dataset
        
        # Recalculate accuracy metrics based on balanced samples
        correct_count = sum(1 for s in balanced_samples if s.get('correct', False))
        true_positives = sum(1 for s in balanced_samples if s.get('correct', False) and s.get('original_did_lie', False))
        true_negatives = sum(1 for s in balanced_samples if s.get('correct', False) and not s.get('original_did_lie', False))
        false_positives = sum(1 for s in balanced_samples if not s.get('correct', False) and not s.get('original_did_lie', False))
        false_negatives = sum(1 for s in balanced_samples if not s.get('correct', False) and s.get('original_did_lie', False))
        
        # Update metrics with balanced values
        metrics['accuracy'] = correct_count / total_samples if total_samples > 0 else 0.0
        metrics['precision'] = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        metrics['recall'] = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0.0
        metrics['tp'] = true_positives
        metrics['tn'] = true_negatives
        metrics['fp'] = false_positives
        metrics['fn'] = false_negatives
        
        # Add balance information
        metrics['balanced_true_count'] = true_count
        metrics['balanced_false_count'] = false_count
        metrics['balance_ratio'] = min(true_count, false_count) / max(true_count, false_count) if max(true_count, false_count) > 0 else 1.0
        
        # Update target and prediction distributions for balanced dataset
        target_a_count = sum(1 for s in balanced_samples if s.get('target') == 'A')
        target_b_count = sum(1 for s in balanced_samples if s.get('target') == 'B')
        pred_a_count = sum(1 for s in balanced_samples if s.get('extracted_prediction') == 'A')
        pred_b_count = sum(1 for s in balanced_samples if s.get('extracted_prediction') == 'B')
        pred_other_count = sum(1 for s in balanced_samples if s.get('extracted_prediction') not in ['A', 'B'])
        
        metrics['target_distribution'] = {
            'A': target_a_count,
            'B': target_b_count
        }
        metrics['prediction_distribution'] = {
            'A': pred_a_count,
            'B': pred_b_count,
            'OTHER': pred_other_count
        }
    
    # Calculate balanced overall metrics
    all_balanced_samples = []
    for subtask_data in balanced_data['subtask_results'].values():
        all_balanced_samples.extend(subtask_data.get('samples', []))
    
    total_balanced_samples = len(all_balanced_samples)
    if total_balanced_samples > 0:
        # Calculate overall metrics from all balanced samples
        correct_count = sum(1 for s in all_balanced_samples if s.get('correct', False))
        true_positives = sum(1 for s in all_balanced_samples if s.get('correct', False) and s.get('original_did_lie', False))
        true_negatives = sum(1 for s in all_balanced_samples if s.get('correct', False) and not s.get('original_did_lie', False))
        false_positives = sum(1 for s in all_balanced_samples if not s.get('correct', False) and not s.get('original_did_lie', False))
        false_negatives = sum(1 for s in all_balanced_samples if not s.get('correct', False) and s.get('original_did_lie', False))
        
        # Update overall metrics with balanced values
        balanced_data['overall_metrics']['parseable_samples'] = total_balanced_samples
        balanced_data['overall_metrics']['filtered_samples'] = 0
        balanced_data['overall_metrics']['accuracy'] = correct_count / total_balanced_samples
        balanced_data['overall_metrics']['precision'] = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        balanced_data['overall_metrics']['recall'] = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        balanced_data['overall_metrics']['f1_score'] = 2 * (balanced_data['overall_metrics']['precision'] * balanced_data['overall_metrics']['recall']) / (balanced_data['overall_metrics']['precision'] + balanced_data['overall_metrics']['recall']) if (balanced_data['overall_metrics']['precision'] + balanced_data['overall_metrics']['recall']) > 0 else 0.0
        balanced_data['overall_metrics']['true_positives'] = true_positives
        balanced_data['overall_metrics']['true_negatives'] = true_negatives
        balanced_data['overall_metrics']['false_positives'] = false_positives
        balanced_data['overall_metrics']['false_negatives'] = false_negatives
    
    # Update overall metadata
    balanced_data['metadata']['balanced_total_samples'] = total_balanced_samples
    
    return balanced_data

def save_balanced_results(balanced_data: Dict[str, Any], output_path: Path):
    """Save balanced results to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(balanced_data, f, indent=2)
    
    print(f"Saved balanced results: {output_path}")

def create_balancing_summary(original_data: Dict[str, Any], balanced_data: Dict[str, Any], 
                           model_name: str, baseline_type: str) -> Dict[str, Any]:
    """Create a summary of balancing statistics for a model."""
    summary = {
        "model": model_name,
        "baseline_type": baseline_type,
        "timestamp": original_data.get('metadata', {}).get('timestamp', ''),
        "subtasks": {}
    }
    
    # Process each subtask
    for subtask_name in original_data.get('subtask_results', {}).keys():
        original_subtask = original_data['subtask_results'][subtask_name]
        balanced_subtask = balanced_data['subtask_results'][subtask_name]
        
        original_samples = original_subtask.get('samples', [])
        balanced_samples = balanced_subtask.get('samples', [])
        
        # Count original samples
        original_true_count = sum(1 for s in original_samples if s.get('original_did_lie', False))
        original_false_count = sum(1 for s in original_samples if not s.get('original_did_lie', False))
        
        # Count balanced samples
        balanced_true_count = sum(1 for s in balanced_samples if s.get('original_did_lie', False))
        balanced_false_count = sum(1 for s in balanced_samples if not s.get('original_did_lie', False))
        
        # Target distributions
        original_target_a = sum(1 for s in original_samples if s.get('target') == 'A')
        original_target_b = sum(1 for s in original_samples if s.get('target') == 'B')
        balanced_target_a = sum(1 for s in balanced_samples if s.get('target') == 'A')
        balanced_target_b = sum(1 for s in balanced_samples if s.get('target') == 'B')
        
        # Prediction distributions
        original_pred_a = sum(1 for s in original_samples if s.get('extracted_prediction') == 'A')
        original_pred_b = sum(1 for s in original_samples if s.get('extracted_prediction') == 'B')
        original_pred_other = sum(1 for s in original_samples if s.get('extracted_prediction') not in ['A', 'B'])
        balanced_pred_a = sum(1 for s in balanced_samples if s.get('extracted_prediction') == 'A')
        balanced_pred_b = sum(1 for s in balanced_samples if s.get('extracted_prediction') == 'B')
        balanced_pred_other = sum(1 for s in balanced_samples if s.get('extracted_prediction') not in ['A', 'B'])
        
        summary["subtasks"][subtask_name] = {
            "original": {
                "total_samples": len(original_samples),
                "did_lie_true": original_true_count,
                "did_lie_false": original_false_count,
                "target_distribution": {
                    "A": original_target_a,
                    "B": original_target_b
                },
                "prediction_distribution": {
                    "A": original_pred_a,
                    "B": original_pred_b,
                    "OTHER": original_pred_other
                }
            },
            "balanced": {
                "total_samples": len(balanced_samples),
                "did_lie_true": balanced_true_count,
                "did_lie_false": balanced_false_count,
                "target_distribution": {
                    "A": balanced_target_a,
                    "B": balanced_target_b
                },
                "prediction_distribution": {
                    "A": balanced_pred_a,
                    "B": balanced_pred_b,
                    "OTHER": balanced_pred_other
                }
            },
            "balancing_info": {
                "was_balanced": len(original_samples) != len(balanced_samples),
                "original_balance_ratio": min(original_true_count, original_false_count) / max(original_true_count, original_false_count) if max(original_true_count, original_false_count) > 0 else 1.0,
                "balanced_balance_ratio": min(balanced_true_count, balanced_false_count) / max(balanced_true_count, balanced_false_count) if max(balanced_true_count, balanced_false_count) > 0 else 1.0,
                "samples_removed": len(original_samples) - len(balanced_samples)
            }
        }
    
    return summary

def save_balancing_summary(summary_data: Dict[str, Any], output_dir: Path, model_name: str, baseline_type: str):
    """Save balancing summary to file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename from model name and baseline type
    model_safe_name = model_name.replace('/', '_').replace(':', '_')
    filename = f"balancing_summary_{model_safe_name}_{baseline_type}.json"
    output_path = output_dir / filename
    
    with open(output_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"Saved balancing summary: {output_path}")

def main():
    """Main function to process all baseline results and create balanced versions."""
    print("Starting post-processing to create balanced datasets...")
    print(f"Source directory: {RESULTS_DIR}")
    print(f"Output directory: {BALANCED_RESULTS_DIR}")
    
    # Set random seed for reproducible sampling
    random.seed(42)
    
    # Process each baseline type
    for baseline_type in BASELINE_TYPES:
        print(f"\n{'='*60}")
        print(f"Processing baseline type: {baseline_type}")
        print(f"{'='*60}")
        
        baseline_dir = Path(RESULTS_DIR) / baseline_type
        balanced_baseline_dir = Path(BALANCED_RESULTS_DIR) / baseline_type
        
        if not baseline_dir.exists():
            print(f"Warning: {baseline_dir} does not exist, skipping...")
            continue
        
        # Find all JSON files for this baseline type
        json_files = list(baseline_dir.glob("*.json"))
        
        if not json_files:
            print(f"No JSON files found in {baseline_dir}")
            continue
        
        print(f"Found {len(json_files)} JSON files to process")
        
        # Process each JSON file
        for json_file in json_files:
            try:
                # Load original data
                with open(json_file, 'r') as f:
                    original_data = json.load(f)
                
                # Process the file
                balanced_data = process_json_file(json_file, baseline_type)
                
                # Create output path
                output_path = balanced_baseline_dir / json_file.name
                
                # Save balanced results
                save_balanced_results(balanced_data, output_path)
                
                # Create and save balancing summary
                model_name = original_data.get('metadata', {}).get('model', 'unknown')
                summary_data = create_balancing_summary(original_data, balanced_data, model_name, baseline_type)
                save_balancing_summary(summary_data, Path(BALANCED_RESULTS_DIR), model_name, baseline_type)
                
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                continue
    
    print(f"\n{'='*60}")
    print("Post-processing complete!")
    print(f"Balanced results saved to: {BALANCED_RESULTS_DIR}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 