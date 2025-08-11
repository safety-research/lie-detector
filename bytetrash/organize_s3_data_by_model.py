#!/usr/bin/env python3
"""
Organized baseline evaluation setup.

This script:
1. Downloads data from S3 into a timestamped directory
2. Organizes data by model in subfolders
3. Creates a mapping JSON for model evaluation
4. Does NOT run evaluations (just sets up the structure)
"""

import boto3
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

def run_aws_command(cmd: List[str]) -> str:
    """Run AWS CLI command and return output."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"AWS command failed: {e}")
        print(f"Error output: {e.stderr}")
        return ""

def get_s3_file_list(s3_uri: str) -> List[str]:
    """Get list of JSONL files from S3."""
    print(f"Getting file list from {s3_uri}...")
    
    output = run_aws_command([
        "aws", "s3", "ls", 
        s3_uri, 
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
    """Organize S3 files by model and download them."""
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
    
    for dir_name, file_list in model_files.items():
        print(f"\nProcessing model: {dir_name}")
        print(f"  Found {len(file_list)} files")
        
        # Create model directory
        model_dir = base_dir / dir_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        downloaded_model_files = []
        
        for file_path in file_list:
            # Download the file
            local_path = model_dir / Path(file_path).name
            cmd = [
                "aws", "s3", "cp", 
                f"s3://dipika-lie-detection-data/{file_path}", 
                str(local_path)
            ]
            
            print(f"  Downloading: {Path(file_path).name}")
            result = run_aws_command(cmd)
            if result != "":
                downloaded_model_files.append(str(local_path))
        
        downloaded_files[dir_name] = downloaded_model_files
        print(f"  Downloaded {len(downloaded_model_files)} files")
    
    return downloaded_files

def create_model_mapping(downloaded_files: Dict[str, List[str]]) -> Dict[str, str]:
    """Create mapping from folder name to evaluation model name."""
    print("Creating model mapping...")
    
    mapping = {}
    
    for folder_name, files in downloaded_files.items():
        if not files:
            continue
            
        # Get the OpenRouter model ID for this folder
        _, model_id = map_model_name_to_openrouter_id(Path(files[0]).name)
        mapping[folder_name] = model_id
        print(f"  {folder_name} -> {model_id}")
    
    return mapping

def save_model_mapping(mapping: Dict[str, str], base_dir: Path):
    """Save model mapping to JSON file."""
    mapping_file = base_dir / "model_mapping.json"
    
    mapping_data = {
        "timestamp": datetime.now().isoformat(),
        "description": "Mapping from folder names to evaluation model names",
        "mapping": mapping
    }
    
    with open(mapping_file, 'w') as f:
        json.dump(mapping_data, f, indent=2)
    
    print(f"Saved model mapping to: {mapping_file}")

def copy_evaluation_files_to_model_dirs(base_dir: Path):
    """Copy evaluation files to each model directory."""
    print("Copying evaluation files to model directories...")
    
    files_to_copy = ["run.py", "evaluate.py", "task.py", "dataset.py", "scorer.py"]
    
    for model_dir in base_dir.iterdir():
        if model_dir.is_dir() and model_dir.name != "temp_counting":
            print(f"  Copying files to {model_dir.name}/")
            
            for file_name in files_to_copy:
                src_path = Path(file_name)
                dst_path = model_dir / file_name
                
                if src_path.exists():
                    # Copy the file
                    with open(src_path, 'r') as f:
                        content = f.read()
                    
                    with open(dst_path, 'w') as f:
                        f.write(content)

def print_directory_structure(base_dir: Path):
    """Print the created directory structure."""
    print(f"\n{'='*80}")
    print(f"CREATED DIRECTORY STRUCTURE")
    print(f"{'='*80}")
    print(f"Base directory: {base_dir}")
    print(f"Model mapping: {base_dir}/model_mapping.json")
    
    print(f"\nModel directories:")
    for item in base_dir.iterdir():
        if item.is_dir() and item.name != "temp_counting":
            file_count = len(list(item.glob("*.jsonl")))
            print(f"  {item.name}/ ({file_count} JSONL files)")
    
    print(f"\nNext steps:")
    print(f"1. Review the model mapping in {base_dir}/model_mapping.json")
    print(f"2. Run evaluation for each model by going to its directory and running 'python run.py'")
    print(f"3. Or create a script to run all evaluations automatically")

def main():
    """Main function."""
    # Configuration
    S3_URI = "s3://dipika-lie-detection-data/processed-data-gemma/"
    
    # Create timestamped directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(f"organized_evaluation_{timestamp}")
    base_dir.mkdir(exist_ok=True)
    
    print(f"Created evaluation directory: {base_dir}")
    
    try:
        # Step 1: Get file list from S3
        files = get_s3_file_list(S3_URI)
        print(f"Found {len(files)} files in S3")
        
        if len(files) == 0:
            print("No files found in S3!")
            return
        
        # Step 2: Organize files by model and download
        downloaded_files = organize_files_by_model(files, base_dir)
        
        # Step 3: Create model mapping
        model_mapping = create_model_mapping(downloaded_files)
        
        # Step 4: Save model mapping
        save_model_mapping(model_mapping, base_dir)
        
        # Step 5: Copy evaluation files to model directories
        copy_evaluation_files_to_model_dirs(base_dir)
        
        # Step 6: Print directory structure
        print_directory_structure(base_dir)
        
        print(f"\n✅ ORGANIZED EVALUATION SETUP COMPLETE!")
        print(f"Directory: {base_dir}")
        print(f"Ready to run evaluations for each model separately")
        
    except Exception as e:
        print(f"❌ ORGANIZED EVALUATION SETUP FAILED: {e}")
        raise

if __name__ == "__main__":
    main() 