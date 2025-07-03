#!/usr/bin/env python3
"""
Script to extract full traces for a specific model from evaluation logs.
Extracts both correct and incorrect answers to create a balanced dataset.
"""

import json
import glob
import os
import argparse
from pathlib import Path
from collections import defaultdict
import random

def get_latest_file_versions(json_dir: str):
    """
    Given a directory with JSON logs, return a list of full paths to the most recent
    version of each unique file ID (based on timestamp prefix in filename).
    """
    path = Path(json_dir)
    assert path.exists() and path.is_dir(), f"Invalid directory: {json_dir}"

    pattern = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{2}-\d{2})_(.*)\.json$")
    file_groups = defaultdict(list)

    for f in path.glob("*.json"):
        match = pattern.search(f.name)
        if not match:
            continue
        timestamp, file_id = match.groups()
        file_groups[file_id].append((timestamp, f.resolve()))

    latest_files = []
    for file_id, files in file_groups.items():
        # Sort by timestamp descending and take the latest
        files.sort(reverse=True)
        latest_files.append(files[0][1])  # keep the Path object (resolved full path)

    return latest_files

def extract_traces_from_logs(log_dir, target_model, max_traces_per_category=100, random_seed=42):
    """
    Extract full traces for a specific model from log files.
    
    Args:
        log_dir: Directory containing log files
        target_model: Model name to filter for (e.g., 'llama-3.3-70b-instruct')
        max_traces_per_category: Maximum number of traces to extract per category (correct/incorrect)
        random_seed: Random seed for reproducible sampling
    
    Returns:
        Dictionary with 'correct' and 'incorrect' traces
    """
    random.seed(random_seed)
    
    # Get latest file versions
    print(f"Getting latest file versions from {log_dir}...")
    log_files = get_latest_file_versions(log_dir)
    print(f"Found {len(log_files)} latest file versions")
    
    traces = {
        'correct': [],
        'incorrect': []
    }
    
    model_variants = [
        target_model,
        f"openrouter/meta-llama/{target_model}",
        f"meta-llama/{target_model}",
        f"meta/{target_model}"
    ]
    
    for log_file_path in log_files:
        try:
            with open(log_file_path, 'r') as f:
                log_data = json.load(f)
            
            # Check if this log contains the target model
            model_found = False
            for sample in log_data.get('samples', []):
                if 'output' in sample and 'model' in sample['output']:
                    sample_model = sample['output']['model']
                    if any(variant in sample_model for variant in model_variants):
                        model_found = True
                        break
            
            if not model_found:
                continue
            
            print(f"Processing {log_file_path.name} (contains {target_model})")
            
            for sample in log_data.get('samples', []):
                # Check if this sample has the target model
                if 'output' not in sample or 'model' not in sample['output']:
                    continue
                
                sample_model = sample['output']['model']
                if not any(variant in sample_model for variant in model_variants):
                    continue
                
                # Check if sample has choices (successful generation)
                if 'choices' not in sample['output'] or len(sample['output']['choices']) == 0:
                    continue
                
                # Check if sample has scores
                if 'scores' not in sample or not sample['scores']:
                    continue
                
                # Extract the trace
                trace = {
                    'sample_id': sample.get('id', 'unknown'),
                    'model': sample_model,
                    'task': log_data.get('eval', {}).get('task', 'unknown'),
                    'task_id': log_data.get('eval', {}).get('task_id', 'unknown'),
                    'input': sample.get('input', ''),
                    'messages': sample.get('messages', []),
                    'output': sample['output'],
                    'target': sample.get('target', ''),
                    'scores': sample['scores'],
                    'metadata': sample.get('metadata', {}),
                    'events': sample.get('events', []),
                    'store': sample.get('store', {})
                }
                
                # Determine if this is correct or incorrect based on scores
                # This logic may need to be adjusted based on your scoring system
                is_correct = False
                
                # Check for common scoring patterns
                for scorer_name, scorer_data in sample['scores'].items():
                    if isinstance(scorer_data, dict) and 'value' in scorer_data:
                        score_value = scorer_data['value']
                        # Adjust this logic based on your scoring system
                        if isinstance(score_value, (int, float)):
                            if score_value >= 1.0:  # Assuming 1.0+ is correct
                                is_correct = True
                                break
                        elif isinstance(score_value, str):
                            if score_value.lower() in ['correct', 'true', '1', 'yes']:
                                is_correct = True
                                break
                
                # Add to appropriate category
                if is_correct:
                    if len(traces['correct']) < max_traces_per_category:
                        traces['correct'].append(trace)
                else:
                    if len(traces['incorrect']) < max_traces_per_category:
                        traces['incorrect'].append(trace)
                
                # Stop if we have enough traces
                if (len(traces['correct']) >= max_traces_per_category and 
                    len(traces['incorrect']) >= max_traces_per_category):
                    break
            
            # Stop if we have enough traces
            if (len(traces['correct']) >= max_traces_per_category and 
                len(traces['incorrect']) >= max_traces_per_category):
                break
                
        except Exception as e:
            print(f"Error processing {log_file_path.name}: {e}")
            continue
    
    # Randomly sample if we have more than requested
    if len(traces['correct']) > max_traces_per_category:
        traces['correct'] = random.sample(traces['correct'], max_traces_per_category)
    
    if len(traces['incorrect']) > max_traces_per_category:
        traces['incorrect'] = random.sample(traces['incorrect'], max_traces_per_category)
    
    return traces

def save_traces(traces, output_dir, model_name):
    """Save traces to JSON files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save all traces
    all_traces = {
        'model': model_name,
        'total_traces': len(traces['correct']) + len(traces['incorrect']),
        'correct_traces': len(traces['correct']),
        'incorrect_traces': len(traces['incorrect']),
        'traces': traces
    }
    
    output_file = os.path.join(output_dir, f"{model_name.replace('/', '_')}_traces.json")
    with open(output_file, 'w') as f:
        json.dump(all_traces, f, indent=2)
    
    # Save separate files for correct and incorrect
    correct_file = os.path.join(output_dir, f"{model_name.replace('/', '_')}_correct_traces.json")
    with open(correct_file, 'w') as f:
        json.dump(traces['correct'], f, indent=2)
    
    incorrect_file = os.path.join(output_dir, f"{model_name.replace('/', '_')}_incorrect_traces.json")
    with open(incorrect_file, 'w') as f:
        json.dump(traces['incorrect'], f, indent=2)
    
    print(f"Saved traces to:")
    print(f"  - {output_file}")
    print(f"  - {correct_file}")
    print(f"  - {incorrect_file}")

def print_summary(traces, model_name):
    """Print summary of extracted traces"""
    print(f"\n=== TRACE EXTRACTION SUMMARY ===")
    print(f"Model: {model_name}")
    print(f"Total traces: {len(traces['correct']) + len(traces['incorrect'])}")
    print(f"Correct traces: {len(traces['correct'])}")
    print(f"Incorrect traces: {len(traces['incorrect'])}")
    
    if traces['correct']:
        print(f"\nSample correct trace ID: {traces['correct'][0]['sample_id']}")
        print(f"Sample correct task: {traces['correct'][0]['task']}")
    
    if traces['incorrect']:
        print(f"\nSample incorrect trace ID: {traces['incorrect'][0]['sample_id']}")
        print(f"Sample incorrect task: {traces['incorrect'][0]['task']}")

def main():
    parser = argparse.ArgumentParser(description='Extract full traces for a specific model')
    parser.add_argument('--log-dir', required=True, help='Directory containing log files')
    parser.add_argument('--model', required=True, help='Model name to extract traces for (e.g., llama-3.3-70b-instruct)')
    parser.add_argument('--output-dir', default='extracted_traces', help='Output directory for traces')
    parser.add_argument('--max-traces', type=int, default=100, help='Maximum traces per category (correct/incorrect)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
    
    args = parser.parse_args()
    
    print(f"Extracting traces for model: {args.model}")
    print(f"Log directory: {args.log_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max traces per category: {args.max_traces}")
    
    traces = extract_traces_from_logs(
        args.log_dir, 
        args.model, 
        max_traces_per_category=args.max_traces,
        random_seed=args.seed
    )
    
    print_summary(traces, args.model)
    save_traces(traces, args.output_dir, args.model)

if __name__ == "__main__":
    import re
    main() 