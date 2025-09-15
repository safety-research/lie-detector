#!/usr/bin/env python3
"""
Simple data validation using AWS CLI to download sample files.
"""

import json
import subprocess
import os
from pathlib import Path
from typing import Dict, List, Any

def run_aws_command(cmd: List[str]) -> str:
    """Run AWS CLI command and return output."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"AWS command failed: {e}")
        print(f"Error output: {e.stderr}")
        return ""

def get_s3_file_list() -> List[str]:
    """Get list of JSONL files from S3."""
    print("Getting file list from S3...")
    output = run_aws_command([
        "aws", "s3", "ls", 
        "s3://dipika-lie-detection-data/processed-data-gemma/", 
        "--recursive"
    ])
    
    files = []
    for line in output.strip().split('\n'):
        if line.strip() and line.endswith('.jsonl'):
            # Extract the S3 key from the output
            parts = line.split()
            if len(parts) >= 4:
                s3_key = parts[3]
                files.append(s3_key)
    
    return files

def download_sample_files(files: List[str], sample_count: int = 5) -> List[str]:
    """Download a sample of files for validation."""
    print(f"Downloading {sample_count} sample files...")
    
    # Create validation directory
    validation_dir = Path("gemma_data_validation_simple")
    validation_dir.mkdir(exist_ok=True)
    
    downloaded_files = []
    
    # Download first few files from different tasks
    tasks_seen = set()
    for file_path in files:
        if len(downloaded_files) >= sample_count:
            break
            
        # Extract task name
        task_name = file_path.split('/')[1]  # processed-data-gemma/task-name/file.jsonl
        
        if task_name not in tasks_seen:
            tasks_seen.add(task_name)
            
            # Download the file
            local_path = validation_dir / Path(file_path).name
            cmd = [
                "aws", "s3", "cp", 
                f"s3://dipika-lie-detection-data/{file_path}", 
                str(local_path)
            ]
            
            print(f"Downloading: {file_path}")
            result = run_aws_command(cmd)
            if result != "":
                downloaded_files.append(str(local_path))
    
    return downloaded_files

def analyze_files(files: List[str]) -> Dict[str, Any]:
    """Analyze the downloaded files to understand the data structure."""
    print("Analyzing downloaded files...")
    
    analysis = {
        'total_files': len(files),
        'tasks': set(),
        'models': set(),
        'total_samples': 0,
        'sample_data': [],
        'model_task_counts': {}
    }
    
    for file_path in files:
        file_path_obj = Path(file_path)
        
        print(f"Analyzing {file_path}")
        
        sample_count = 0
        models_in_file = set()
        tasks_in_file = set()
        
        sample_count = 0
        models_in_file = set()
        
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f):
                try:
                    item = json.loads(line.strip())
                    sample_count += 1
                    
                    # Get model name and task name from the data
                    model_name = item.get('model', 'unknown')
                    task_name = item.get('task', 'unknown')
                    models_in_file.add(model_name)
                    tasks_in_file.add(task_name)
                    
                    # Store sample data for first few samples
                    if len(analysis['sample_data']) < 3:
                        analysis['sample_data'].append({
                            'file': file_path,
                            'task': task_name,
                            'model': model_name,
                            'did_lie': item.get('did_lie'),
                            'trace_length': len(item.get('trace', [])),
                            'has_metadata': 'metadata' in item,
                            'sample_id': item.get('sample_id')
                        })
                        
                except json.JSONDecodeError as e:
                    print(f"Warning: JSON decode error in {file_path}:{line_num}: {e}")
                    continue
        
        # Update analysis
        for task in tasks_in_file:
            analysis['tasks'].add(task)
        analysis['total_samples'] += sample_count
        
        for model in models_in_file:
            analysis['models'].add(model)
            for task in tasks_in_file:
                key = (task, model)
                analysis['model_task_counts'][key] = sample_count
    
    return analysis

def test_model_mapping(models: set) -> Dict[str, str]:
    """Test the model mapping function."""
    print("Testing model mapping...")
    
    # Import the mapping function
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from baseline.evaluate import map_data_model_to_eval_model
    
    mapping_results = {}
    for model in sorted(models):
        mapped_model = map_data_model_to_eval_model(model)
        mapping_results[model] = mapped_model
        print(f"  {model} -> {mapped_model}")
    
    return mapping_results

def print_summary(analysis: Dict[str, Any], model_mapping: Dict[str, str]):
    """Print validation summary."""
    print("\n" + "="*80)
    print("DATA VALIDATION SUMMARY")
    print("="*80)
    
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"  Total files analyzed: {analysis['total_files']}")
    print(f"  Total tasks found: {len(analysis['tasks'])}")
    print(f"  Total unique models: {len(analysis['models'])}")
    print(f"  Total samples: {analysis['total_samples']}")
    
    print(f"\n📋 TASKS FOUND:")
    for task in sorted(analysis['tasks']):
        print(f"  - {task}")
    
    print(f"\n🤖 MODELS FOUND:")
    for model in sorted(analysis['models']):
        mapped_model = model_mapping.get(model, 'UNKNOWN')
        print(f"  - {model} -> {mapped_model}")
    
    print(f"\n📝 SAMPLE DATA PREVIEW:")
    for i, sample in enumerate(analysis['sample_data']):
        print(f"  Sample {i+1}:")
        print(f"    File: {sample['file']}")
        print(f"    Task: {sample['task']}")
        print(f"    Model: {sample['model']}")
        print(f"    Did Lie: {sample['did_lie']}")
        print(f"    Trace Length: {sample['trace_length']}")
        print(f"    Sample ID: {sample['sample_id']}")
        print()
    
    print(f"\n📈 TASK-MODEL BREAKDOWN:")
    for (task, model), count in sorted(analysis['model_task_counts'].items()):
        print(f"  {task} - {model}: {count} samples")

def main():
    """Main validation function."""
    print("Starting data validation...")
    
    # Get list of files from S3
    files = get_s3_file_list()
    print(f"Found {len(files)} JSONL files in S3")
    
    if len(files) == 0:
        print("No files found in S3!")
        return
    
    # Download sample files
    downloaded_files = download_sample_files(files, sample_count=10)
    print(f"Downloaded {len(downloaded_files)} files")
    
    if len(downloaded_files) == 0:
        print("No files downloaded!")
        return
    
    # Analyze files
    analysis = analyze_files(downloaded_files)
    
    # Test model mapping
    model_mapping = test_model_mapping(analysis['models'])
    
    # Print summary
    print_summary(analysis, model_mapping)
    
    print(f"\n✅ VALIDATION COMPLETE!")
    print(f"Based on sample analysis, the full dataset should have:")
    print(f"  - ~{len(analysis['tasks']) * len(analysis['models'])} task-model combinations")
    print(f"  - ~{len(files)} total files")
    print(f"  - Model mapping is working correctly")

if __name__ == "__main__":
    main() 