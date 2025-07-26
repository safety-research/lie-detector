#!/usr/bin/env python3
"""
Organized baseline evaluation setup using boto3.

This script:
1. Downloads data from S3 into a timestamped directory
2. Organizes data by model in subfolders
3. Creates a mapping JSON for model evaluation
4. Does NOT run evaluations (just sets up the structure)
"""

import boto3
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

def get_s3_file_list(s3_uri: str) -> List[str]:
    """Get list of JSONL files from S3 using boto3."""
    print(f"Getting file list from {s3_uri}...")
    
    # Parse S3 URI
    if s3_uri.startswith('s3://'):
        s3_uri = s3_uri[5:]
    
    bucket_name, prefix = s3_uri.split('/', 1) if '/' in s3_uri else (s3_uri, '')
    
    s3_client = boto3.client('s3')
    files = []
    
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
        
        for page in page_iterator:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if key.endswith('.jsonl'):
                        files.append(key)
        
        print(f"Found {len(files)} JSONL files in S3")
        return files
    except Exception as e:
        print(f"Error listing S3 objects: {e}")
        return []

def map_model_name_to_openrouter_id(filename: str) -> Tuple[str, str]:
    """Map model name from filename to directory name and OpenRouter model ID."""
    filename_lower = filename.lower()
    
    if 'gemma' in filename_lower and '4b-it' in filename_lower:
        return "openrouter_google_gemma-3-4b-it", "openrouter/google/gemma-3-4b-it"
    elif 'gemma' in filename_lower and '12b-it' in filename_lower:
        return "openrouter_google_gemma-3-12b-it", "openrouter/google/gemma-3-12b-it"
    elif 'gemma' in filename_lower and '27b-it' in filename_lower:
        return "openrouter_google_gemma-3-27b-it", "openrouter/google/gemma-3-27b-it"
    else:
        return "unknown", "unknown"

def organize_files_by_model(files: List[str], base_dir: Path) -> Dict[str, List[str]]:
    """Organize S3 files by model and download them using boto3."""
    print("Organizing files by model...")
    
    model_files = {}
    
    for file_path in files:
        filename = Path(file_path).name
        dir_name, model_id = map_model_name_to_openrouter_id(filename)
        
        if dir_name not in model_files:
            model_files[dir_name] = []
        model_files[dir_name].append(file_path)
    
    # Download files for each model
    downloaded_files = {}
    s3_client = boto3.client('s3')
    
    for dir_name, file_list in model_files.items():
        print(f"\nProcessing model: {dir_name}")
        print(f"  Found {len(file_list)} files")
        
        # Create model directory
        model_dir = base_dir / dir_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        downloaded_model_files = []
        
        for file_path in file_list:
            # Download the file using boto3
            local_path = model_dir / Path(file_path).name
            try:
                s3_client.download_file('dipika-lie-detection-data', file_path, str(local_path))
                downloaded_model_files.append(str(local_path))
                print(f"    Downloaded: {Path(file_path).name}")
            except Exception as e:
                print(f"    Error downloading {file_path}: {e}")
        
        downloaded_files[dir_name] = downloaded_model_files
    
    return downloaded_files

def create_model_mapping(downloaded_files: Dict[str, List[str]]) -> Dict[str, str]:
    """Create mapping from directory names to OpenRouter model IDs."""
    mapping = {}
    
    for dir_name in downloaded_files.keys():
        if dir_name == "openrouter_google_gemma-3-4b-it":
            mapping[dir_name] = "openrouter/google/gemma-3-4b-it"
        elif dir_name == "openrouter_google_gemma-3-12b-it":
            mapping[dir_name] = "openrouter/google/gemma-3-12b-it"
        elif dir_name == "openrouter_google_gemma-3-27b-it":
            mapping[dir_name] = "openrouter/google/gemma-3-27b-it"
        else:
            mapping[dir_name] = "unknown"
    
    return mapping

def save_model_mapping(mapping: Dict[str, str], base_dir: Path):
    """Save model mapping to JSON file."""
    mapping_file = base_dir / "model_mapping.json"
    
    with open(mapping_file, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"\nModel mapping saved to: {mapping_file}")

def copy_evaluation_files_to_model_dirs(base_dir: Path):
    """Copy evaluation files to each model directory."""
    print("\nCopying evaluation files to model directories...")
    
    # List of evaluation files to copy
    eval_files = [
        "run_cleanup_sample_id.py",
        "run_test_simple.py",
        "test_consistency_scorer.py",
        "test_contrastive.py",
        "test_cot_hook.py",
        "test_cot_unfaithfulness.py",
        "test_elicitation_classifier.py"
    ]
    
    for model_dir in base_dir.iterdir():
        if model_dir.is_dir() and model_dir.name != "__pycache__":
            print(f"  Copying files to {model_dir.name}/")
            
            for eval_file in eval_files:
                source_path = Path(eval_file)
                if source_path.exists():
                    dest_path = model_dir / eval_file
                    try:
                        with open(source_path, 'r') as src, open(dest_path, 'w') as dst:
                            dst.write(src.read())
                        print(f"    Copied: {eval_file}")
                    except Exception as e:
                        print(f"    Error copying {eval_file}: {e}")

def print_directory_structure(base_dir: Path):
    """Print the organized directory structure."""
    print(f"\n📁 Organized Evaluation Directory Structure:")
    print(f"   {base_dir}")
    
    for model_dir in sorted(base_dir.iterdir()):
        if model_dir.is_dir() and model_dir.name != "__pycache__":
            print(f"   └── {model_dir.name}/")
            
            # Count files in model directory
            file_count = len([f for f in model_dir.iterdir() if f.is_file()])
            print(f"       ({file_count} files)")
    
    print(f"\n📄 Model mapping: {base_dir}/model_mapping.json")

def main():
    """Main function."""
    S3_URI = "s3://dipika-lie-detection-data/processed-data-gemma/"
    
    # Create timestamped evaluation directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    evaluation_dir = Path(f"organized_evaluation_{timestamp}")
    evaluation_dir.mkdir(exist_ok=True)
    
    print(f"Created evaluation directory: {evaluation_dir}")
    
    # Get file list from S3
    files = get_s3_file_list(S3_URI)
    
    if not files:
        print("No files found in S3!")
        return
    
    # Organize and download files by model
    downloaded_files = organize_files_by_model(files, evaluation_dir)
    
    if not downloaded_files:
        print("No files were downloaded!")
        return
    
    # Create and save model mapping
    mapping = create_model_mapping(downloaded_files)
    save_model_mapping(mapping, evaluation_dir)
    
    # Copy evaluation files
    copy_evaluation_files_to_model_dirs(evaluation_dir)
    
    # Print directory structure
    print_directory_structure(evaluation_dir)
    
    print(f"\n✅ Organized evaluation setup complete!")
    print(f"   Ready to run evaluations in: {evaluation_dir}")

if __name__ == "__main__":
    main() 