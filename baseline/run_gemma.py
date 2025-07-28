#!/usr/bin/env python3
"""
Script to run evaluation on organized Gemma model directories.
Reads the model mapping JSON and runs evaluation for each directory using the specified evaluation model.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add the parent directory to Python path to ensure baseline module can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Configuration
ORGANIZED_DIR = "/Users/dipikakhullar/Desktop/lie-detector/baseline/organized_evaluation_20250721_215718"
NUM_SAMPLES = None  # Set to a number to limit samples, or None for all
# BASELINE_TYPES = ["base_transcript", "rowans_escaped_transcript", "llama_chat", "llama_chat_reasoning", "base_transcript_reasoning"]
BASELINE_TYPES = ["base_transcript", "rowans_escaped_transcript", "llama_chat", "llama_chat_reasoning"]

def load_model_mapping(organized_dir: Path) -> dict:
    """Load the model mapping from the JSON file."""
    mapping_file = organized_dir / "model_mapping.json"
    
    if not mapping_file.exists():
        raise FileNotFoundError(f"Model mapping file not found: {mapping_file}")
    
    with open(mapping_file, 'r') as f:
        mapping_data = json.load(f)
    
    return mapping_data["mapping"]





def run_evaluation_for_model(model_folder: str, eval_model: str, organized_dir: Path):
    """Run evaluation for a specific model."""
    print(f"\n{'='*80}")
    print(f"🚀 RUNNING EVALUATION FOR: {model_folder}")
    print(f"🎯 EVALUATION MODEL: {eval_model}")
    print(f"{'='*80}")
    
    model_dir = organized_dir / model_folder
    
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        return False
    
    # Change to the model directory for evaluation
    original_cwd = os.getcwd()
    os.chdir(model_dir)
    
    try:
        # Import evaluation functions
        from evaluate import main_by_model
        
        print(f"📁 Running evaluation from: {model_dir}")
        
        # Set up top-level results directories
        organized_dir = Path(ORGANIZED_DIR)
        top_level_results = organized_dir / "results"
        top_level_transcripts = organized_dir / "transcripts"
        
        # Create directories if they don't exist
        top_level_results.mkdir(exist_ok=True)
        top_level_transcripts.mkdir(exist_ok=True)
        
        # Run evaluation for each baseline type
        success = True
        for baseline_type in BASELINE_TYPES:
            print(f"\n{'='*40}\nRunning baseline: {baseline_type}\n{'='*40}")
            
            try:
                # Create baseline-specific directories
                baseline_results_dir = top_level_results / baseline_type
                baseline_transcripts_dir = top_level_transcripts / baseline_type
                baseline_results_dir.mkdir(exist_ok=True)
                baseline_transcripts_dir.mkdir(exist_ok=True)
                
                # Temporarily change working directory to model directory for data access
                os.chdir(model_dir)
                
                # Run the evaluation using the imported function with custom output directories
                main_by_model(
                    num_samples=NUM_SAMPLES, 
                    model=eval_model, 
                    data_dir=".", 
                    baseline_type=baseline_type, 
                    use_same_model=False,
                    results_dir=str(baseline_results_dir),
                    transcripts_dir=str(baseline_transcripts_dir),
                    model_prefix=model_folder
                )
                
                print(f"✅ {baseline_type} completed successfully")
            except Exception as e:
                print(f"❌ {baseline_type} failed: {e}")
                success = False
        
        if success:
            print(f"✅ All evaluations completed successfully for {model_folder}")
            return True
        else:
            print(f"❌ Some evaluations failed for {model_folder}")
            return False
            
    except Exception as e:
        print(f"❌ Error running evaluation for {model_folder}: {e}")
        return False
    
    finally:
        os.chdir(original_cwd)

def main():
    """Main function to run evaluation on all organized model directories."""
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
    
    print(f"\n⚙️ Configuration:")
    print(f"  NUM_SAMPLES: {NUM_SAMPLES}")
    print(f"  BASELINE_TYPES: {BASELINE_TYPES}")
    print(f"  EVALUATION MODE: Use model from model_mapping.json for each directory")
    
    # Run evaluation for each model
    successful_runs = 0
    total_runs = len(model_mapping)
    
    for folder_name, eval_model in model_mapping.items():
        if eval_model == "unknown":
            print(f"\n⏭️ Skipping {folder_name} - unknown model")
            continue
        
        if run_evaluation_for_model(folder_name, eval_model, organized_dir):
            successful_runs += 1
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"📊 EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total models: {total_runs}")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {total_runs - successful_runs}")
    
    if successful_runs == total_runs:
        print(f"🎉 All evaluations completed successfully!")
    else:
        print(f"⚠️ Some evaluations failed. Check the logs above.")
    
    # Show results location
    print(f"\n📁 Results are organized by baseline type:")
    print(f"  Results: {ORGANIZED_DIR}/results/[baseline_type]/")
    print(f"  Transcripts: {ORGANIZED_DIR}/transcripts/[baseline_type]/")
    
    print(f"{'='*80}")

if __name__ == "__main__":
    main() 