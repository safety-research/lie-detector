#!/usr/bin/env python3
"""
Script to count total samples for each model across all datasets.
Outputs results in CSV format.
"""

import boto3
import json
import csv
from collections import defaultdict

def count_samples_by_model():
    """Count total samples for each model across all datasets."""
    s3 = boto3.client('s3')
    bucket = "dipika-lie-detection-data"
    prefix = "processed-data-gemma/"
    
    # Get all deduplicated files
    paginator = s3.get_paginator('list_objects_v2')
    model_counts = defaultdict(int)
    dataset_counts = defaultdict(lambda: defaultdict(int))
    csv_data = []
    
    print("Counting samples for each model...")
    print("=" * 60)
    
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if 'deduplicated' in key and key.endswith('.jsonl'):
                # Extract model name from file path
                # Example: processed-data-gemma/alibi-fraud-investigation/deduplicated_alibi-fraud-investigation_google_gemma-3-27b-it.jsonl
                parts = key.split('/')
                if len(parts) >= 3:
                    dataset = parts[1]  # e.g., "alibi-fraud-investigation"
                    filename = parts[2]  # e.g., "deduplicated_alibi-fraud-investigation_google_gemma-3-27b-it.jsonl"
                    
                    # Extract model name from filename
                    if '_openrouter_' in filename:
                        model = filename.split('_openrouter_')[1].replace('.jsonl', '')
                    elif '_google_' in filename:
                        model = filename.split('_google_')[1].replace('.jsonl', '')
                    elif '_meta-llama_' in filename:
                        model = filename.split('_meta-llama_')[1].replace('.jsonl', '')
                    else:
                        # Fallback: extract everything after the last underscore
                        model = filename.split('_')[-1].replace('.jsonl', '')
                    
                    # Count samples in this file
                    try:
                        response = s3.get_object(Bucket=bucket, Key=key)
                        content = response['Body'].read().decode('utf-8')
                        lines = content.strip().split('\n')
                        sample_count = len([line for line in lines if line.strip()])
                        
                        model_counts[model] += sample_count
                        dataset_counts[dataset][model] = sample_count
                        csv_data.append([dataset, model, sample_count])
                        
                        print(f"{dataset:40} | {model:35} | {sample_count:6} samples")
                        
                    except Exception as e:
                        print(f"Error reading {key}: {e}")
    
    print("\n" + "=" * 60)
    print("TOTAL SAMPLES BY MODEL:")
    print("=" * 60)
    
    total_samples = 0
    for model, count in sorted(model_counts.items()):
        print(f"{model:35} | {count:6} samples")
        total_samples += count
    
    print("=" * 60)
    print(f"{'TOTAL':35} | {total_samples:6} samples")
    
    # Write CSV file
    csv_filename = "sample_counts.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['task_name', 'model', 'num_samples'])
        for row in sorted(csv_data):
            writer.writerow(row)
    
    print(f"\nCSV file written to: {csv_filename}")
    print(f"Total rows: {len(csv_data)}")

if __name__ == "__main__":
    count_samples_by_model() 