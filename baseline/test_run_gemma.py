#!/usr/bin/env python3
"""
Test script to validate the run_gemma setup without running full evaluations.
"""

import json
import os
import sys
from pathlib import Path

# Configuration
ORGANIZED_DIR = "/Users/dipikakhullar/Desktop/lie-detector/baseline/organized_evaluation_20250721_215718"
NUM_SAMPLES = 10  # Set to a number to limit samples, or None for all
BASELINE_TYPES = ["base_transcript", "rowans_escaped_transcript", "llama_chat", "llama_chat_reasoning", "base_transcript_reasoning"]

def load_model_mapping(organized_dir: Path) -> dict:
    """Load the model mapping from the JSON file."""
    mapping_file = organized_dir / "model_mapping.json"
    
    if not mapping_file.exists():
        raise FileNotFoundError(f"Model mapping file not found: {mapping_file}")
    
    with open(mapping_file, 'r') as f:
        mapping_data = json.load(f)
    
    return mapping_data["mapping"]

def update_run_py_config(model_dir: Path, eval_model: str, baseline_types: list, num_samples: int | None = None):
    """Update the run.py configuration in a model directory."""
    run_py_path = model_dir / "run.py"
    
    if not run_py_path.exists():
        print(f"Warning: run.py not found in {model_dir}")
        return False
    
    # Read the current run.py
    with open(run_py_path, 'r') as f:
        content = f.read()
    
    # Update the configuration
    # Update MODEL_NAME
    content = content.replace(
        'MODEL_NAME = "openrouter/google/gemma-3-12b-it"',
        f'MODEL_NAME = "{eval_model}"'
    )
    
    # Update BASELINE_TYPES - handle different possible current values
    baseline_types_str = '[' + ', '.join([f'"{bt}"' for bt in baseline_types]) + ']'
    
    # Try different possible current BASELINE_TYPES patterns
    baseline_patterns = [
        'BASELINE_TYPES = ["escaped_transcript", "rowans_escaped_transcript", "llama_chat"]',
        'BASELINE_TYPES = ["escaped_transcript"]',
        'BASELINE_TYPES = ["base_transcript", "rowans_escaped_transcript", "llama_chat", "llama_chat_reasoning", "base_transcript_reasoning"]'
    ]
    
    baseline_updated = False
    for pattern in baseline_patterns:
        if pattern in content:
            content = content.replace(pattern, f'BASELINE_TYPES = {baseline_types_str}')
            baseline_updated = True
            break
    
    if not baseline_updated:
        # If no pattern matched, try to find and replace any BASELINE_TYPES line
        import re
        content = re.sub(
            r'BASELINE_TYPES = \[.*?\]',
            f'BASELINE_TYPES = {baseline_types_str}',
            content
        )
    
    # Update NUM_SAMPLES if specified - handle different possible current values
    if num_samples is not None:
        num_samples_patterns = [
            'NUM_SAMPLES = None',
            'NUM_SAMPLES = 2',
            'NUM_SAMPLES = 10'
        ]
        
        num_samples_updated = False
        for pattern in num_samples_patterns:
            if pattern in content:
                content = content.replace(pattern, f'NUM_SAMPLES = {num_samples}')
                num_samples_updated = True
                break
        
        if not num_samples_updated:
            # If no pattern matched, try to find and replace any NUM_SAMPLES line
            import re
            content = re.sub(
                r'NUM_SAMPLES = \d+',
                f'NUM_SAMPLES = {num_samples}',
                content
            )
    
    # Update S3_URI to None (use local data)
    content = content.replace(
        'S3_URI = "s3://dipika-lie-detection-data/processed-data-gemma/"',
        'S3_URI = None'
    )
    
    # Set USE_SAME_MODEL = False (simplified approach)
    content = content.replace(
        'USE_SAME_MODEL = True',
        'USE_SAME_MODEL = False'
    )
    
    # Write the updated run.py
    with open(run_py_path, 'w') as f:
        f.write(content)
    
    print(f"  ✅ Updated run.py configuration for {eval_model}")
    return True

def validate_model_directory(model_folder: str, eval_model: str, organized_dir: Path):
    """Validate a model directory without running evaluation."""
    print(f"\n{'='*60}")
    print(f"🔍 VALIDATING: {model_folder}")
    print(f"🎯 EVALUATION MODEL: {eval_model}")
    print(f"{'='*60}")
    
    model_dir = organized_dir / model_folder
    
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        return False
    
    # Check for required files
    required_files = ["run.py", "evaluate.py", "task.py", "dataset.py", "scorer.py"]
    missing_files = []
    
    for file_name in required_files:
        file_path = model_dir / file_name
        if not file_path.exists():
            missing_files.append(file_name)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    # Count JSONL files
    jsonl_files = list(model_dir.glob("*.jsonl"))
    print(f"📁 Found {len(jsonl_files)} JSONL files")
    
    # Update run.py configuration for this model
    if not update_run_py_config(model_dir, eval_model, BASELINE_TYPES, NUM_SAMPLES):
        return False
    
    print(f"✅ Directory validation successful for {model_folder}")
    return True

def main():
    """Main function to validate all organized model directories."""
    organized_dir = Path(ORGANIZED_DIR)
    
    if not organized_dir.exists():
        print(f"❌ Organized directory not found: {organized_dir}")
        return
    
    print(f"🔍 Loading model mapping from: {organized_dir}/model_mapping.json")
    
    try:
        model_mapping = load_model_mapping(organized_dir)
    except Exception as e:
        print(f"❌ Error loading model mapping: {e}")
        return
    
    print(f"📋 Model mapping loaded successfully:")
    for folder, model in model_mapping.items():
        print(f"  {folder} -> {model}")
    
    print(f"\n⚙️ Test Configuration:")
    print(f"  NUM_SAMPLES: {NUM_SAMPLES}")
    print(f"  BASELINE_TYPES: {BASELINE_TYPES}")
    print(f"  EVALUATION MODE: Use model from model_mapping.json for each directory")
    
    # Validate each model directory
    successful_validations = 0
    total_validations = len(model_mapping)
    
    for folder_name, eval_model in model_mapping.items():
        if eval_model == "unknown":
            print(f"\n⏭️ Skipping {folder_name} - unknown model")
            continue
        
        if validate_model_directory(folder_name, eval_model, organized_dir):
            successful_validations += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"📊 VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total models: {total_validations}")
    print(f"Successful validations: {successful_validations}")
    print(f"Failed validations: {total_validations - successful_validations}")
    
    if successful_validations == total_validations:
        print(f"🎉 All directories validated successfully!")
        print(f"💡 Ready to run full evaluation with: python run_gemma.py")
    else:
        print(f"⚠️ Some validations failed. Check the logs above.")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 