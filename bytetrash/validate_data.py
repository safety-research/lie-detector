#!/usr/bin/env python3
"""
Validate data before running baseline evaluation.

This script checks:
1. How many tasks/samples per model will be loaded
2. If model mapping functions work correctly
3. Data structure validation
"""

import boto3
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
import os
import sys

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from baseline.evaluate import map_data_model_to_eval_model

def copy_data_from_s3(s3_uri: str, local_dir: str) -> str:
    """Copy data from S3 to local directory."""
    print(f"============================================================")
    print(f"Using S3 data source: {s3_uri}")
    print(f"============================================================")
    
    # Parse S3 URI
    if not s3_uri.startswith('s3://'):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    
    bucket_name = s3_uri.split('/')[2]
    prefix = '/'.join(s3_uri.split('/')[3:]) + '/'
    
    print(f"Listing objects in {s3_uri} ...")
    
    # Create local directory
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize S3 client
    s3_client = boto3.client('s3')
    
    # List objects in S3
    paginator = s3_client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    
    downloaded_files = []
    
    file_count = 0
    for page in page_iterator:
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                file_count += 1
                if file_count <= 5:  # Debug: show first 5 objects
                    print(f"Found object: {key}")
                if key.endswith('.jsonl'):
                    # Create local file path
                    relative_path = key[len(prefix):]
                    local_file_path = local_path / relative_path
                    local_file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    print(f"Downloading s3://{bucket_name}/{key} -> {local_file_path}")
                    try:
                        s3_client.download_file(bucket_name, key, str(local_file_path))
                        downloaded_files.append(str(local_file_path))
                    except Exception as e:
                        print(f"Error downloading {key}: {e}")
                        continue
    
    print(f"Downloaded {len(downloaded_files)} files")
    return str(local_path)

def load_and_validate_data(data_dir: str) -> Dict[str, Any]:
    """Load and validate data from the specified directory."""
    print(f"\n============================================================")
    print(f"LOADING AND VALIDATING DATA FROM: {data_dir}")
    print(f"============================================================")
    
    data_dir_path = Path(data_dir)
    if not data_dir_path.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    # Find all JSONL files
    jsonl_files = list(data_dir_path.rglob("*.jsonl"))
    print(f"Found {len(jsonl_files)} JSONL files")
    
    # Group by model and task
    model_task_counts = {}
    model_counts = {}
    task_counts = {}
    sample_data = {}
    
    for file_path in jsonl_files:
        # Extract task name from path
        task_name = file_path.parent.name
        
        # Count samples and get model info
        sample_count = 0
        models_in_file = set()
        
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f):
                try:
                    item = json.loads(line.strip())
                    sample_count += 1
                    
                    # Get model name
                    model_name = item.get('model', 'unknown')
                    models_in_file.add(model_name)
                    
                    # Store sample data for validation
                    if len(sample_data) < 5:  # Store first 5 samples for inspection
                        sample_data[f"{task_name}_{line_num}"] = {
                            'task': task_name,
                            'model': model_name,
                            'did_lie': item.get('did_lie'),
                            'trace_length': len(item.get('trace', [])),
                            'has_metadata': 'metadata' in item
                        }
                        
                except json.JSONDecodeError as e:
                    print(f"Warning: JSON decode error in {file_path}:{line_num}: {e}")
                    continue
        
        # Update counts
        for model in models_in_file:
            key = (task_name, model)
            model_task_counts[key] = sample_count
            
            model_counts[model] = model_counts.get(model, 0) + sample_count
            task_counts[task_name] = task_counts.get(task_name, 0) + sample_count
    
    return {
        'model_task_counts': model_task_counts,
        'model_counts': model_counts,
        'task_counts': task_counts,
        'sample_data': sample_data,
        'total_files': len(jsonl_files)
    }

def validate_model_mapping(data_info: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that model mapping works correctly."""
    print(f"\n============================================================")
    print(f"VALIDATING MODEL MAPPING")
    print(f"============================================================")
    
    model_mapping_results = {}
    
    # Test model mapping for each unique model in the data
    unique_models = set()
    for (task, model) in data_info['model_task_counts'].keys():
        unique_models.add(model)
    
    print(f"Found {len(unique_models)} unique models in data:")
    for model in sorted(unique_models):
        mapped_model = map_data_model_to_eval_model(model)
        model_mapping_results[model] = mapped_model
        print(f"  {model} -> {mapped_model}")
    
    return model_mapping_results

def print_summary(data_info: Dict[str, Any], model_mapping: Dict[str, str]):
    """Print a comprehensive summary of the data."""
    print(f"\n============================================================")
    print(f"DATA VALIDATION SUMMARY")
    print(f"============================================================")
    
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"  Total files: {data_info['total_files']}")
    print(f"  Total tasks: {len(data_info['task_counts'])}")
    print(f"  Total unique models: {len(data_info['model_counts'])}")
    print(f"  Total samples: {sum(data_info['model_counts'].values())}")
    
    print(f"\n📋 TASKS ({len(data_info['task_counts'])} total):")
    for task, count in sorted(data_info['task_counts'].items()):
        print(f"  {task}: {count} samples")
    
    print(f"\n🤖 MODELS ({len(data_info['model_counts'])} total):")
    for model, count in sorted(data_info['model_counts'].items()):
        mapped_model = model_mapping.get(model, 'UNKNOWN')
        print(f"  {model} -> {mapped_model}: {count} samples")
    
    print(f"\n📝 SAMPLE DATA PREVIEW:")
    for key, sample in data_info['sample_data'].items():
        print(f"  {key}:")
        print(f"    Task: {sample['task']}")
        print(f"    Model: {sample['model']}")
        print(f"    Did Lie: {sample['did_lie']}")
        print(f"    Trace Length: {sample['trace_length']}")
        print(f"    Has Metadata: {sample['has_metadata']}")
        print()

def main():
    """Main validation function."""
    # Configuration
    S3_URI = "s3://dipika-lie-detection-data/processed-data-gemma/"
    LOCAL_DIR = "gemma_data_validation"
    
    try:
        # Copy data from S3
        data_dir = copy_data_from_s3(S3_URI, LOCAL_DIR)
        
        # Load and validate data
        data_info = load_and_validate_data(data_dir)
        
        # Validate model mapping
        model_mapping = validate_model_mapping(data_info)
        
        # Print summary
        print_summary(data_info, model_mapping)
        
        print(f"\n✅ VALIDATION COMPLETE!")
        print(f"Data directory: {data_dir}")
        print(f"Ready to run evaluation with {len(data_info['task_counts'])} tasks and {len(data_info['model_counts'])} models")
        
    except Exception as e:
        print(f"❌ VALIDATION FAILED: {e}")
        raise

if __name__ == "__main__":
    main() 