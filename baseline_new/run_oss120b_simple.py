#!/usr/bin/env python3
"""
Simple script to run evaluation on OSS 120B data by creating temporary model directories.
This works around the model inference issue by creating the expected directory structure.
"""

import json
import os
import sys
import shutil
import tempfile
from pathlib import Path
from datetime import datetime

# Set the log directory to be within baseline_new instead of .data
current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["INSPECT_LOG_DIR"] = os.path.join(current_dir, "logs")

# Add the parent directory to Python path to ensure baseline module can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import S3 update functionality
from utils import post_process_results

# Configuration
ORGANIZED_DIR = "/Users/dipikakhullar/Desktop/lie_detector_clone/lie-detector/.data/openai/gpt_oss_120b"
NUM_SAMPLES = 100  # Set to a number to limit samples, or None for all
BASELINE_TYPES = ["llama_chat", "base_transcript", "rowans_escaped_transcript", "llama_chat_reasoning", "rowans_escaped_transcript_reasoning"]
EVALUATION_MODEL = "openrouter/openai/gpt-oss-120b"  # Model to use for evaluation
UPDATE_S3 = True  # Set to True to automatically update S3 files with baseline results

def create_temp_model_structure(organized_dir: Path, category_name: str) -> Path:
    """Create a temporary directory structure that the evaluation system expects."""
    # Create a temporary directory
    temp_dir = Path(tempfile.mkdtemp(prefix=f"oss120b_{category_name}_"))
    
    # Create the expected model directory structure
    model_dir = temp_dir / "gpt_oss_120b"  # This will be recognized as a model
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy the train.jsonl and val.jsonl files
    source_train = organized_dir / category_name / "train.jsonl"
    source_val = organized_dir / category_name / "val.jsonl"
    
    if source_train.exists():
        shutil.copy2(source_train, model_dir / "train.jsonl")
    if source_val.exists():
        shutil.copy2(source_val, model_dir / "val.jsonl")
    
    return temp_dir

def run_evaluation_for_category(category_name: str, eval_model: str, organized_dir: Path):
    """Run evaluation for a specific category."""
    print(f"\n{'='*80}")
    print(f"🚀 RUNNING EVALUATION FOR CATEGORY: {category_name}")
    print(f"🎯 EVALUATION MODEL: {eval_model}")
    print(f"{'='*80}")
    
    category_dir = organized_dir / category_name
    
    if not category_dir.exists():
        print(f"❌ Category directory not found: {category_dir}")
        return False
    
    # Check if train.jsonl and val.jsonl exist
    train_file = category_dir / "train.jsonl"
    val_file = category_dir / "val.jsonl"
    
    if not train_file.exists() or not val_file.exists():
        print(f"❌ Missing train.jsonl or val.jsonl in {category_dir}")
        return False
    
    # Create temporary directory structure
    temp_dir = create_temp_model_structure(organized_dir, category_name)
    print(f"📁 Created temporary structure: {temp_dir}")
    
    # Change to the temp directory for evaluation
    original_cwd = os.getcwd()
    os.chdir(temp_dir)
    
    try:
        # Import evaluation functions
        from evaluate import main_by_model
        
        print(f"📁 Running evaluation from: {temp_dir}")
        
        # Set up top-level results directories
        baseline_dir = Path(__file__).parent
        top_level_results = baseline_dir / "results"
        top_level_transcripts = baseline_dir / "transcripts"
        
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
                
                # Create category-specific subdirectories
                category_results_dir = baseline_results_dir / category_name
                category_transcripts_dir = baseline_transcripts_dir / category_name
                category_results_dir.mkdir(exist_ok=True)
                category_transcripts_dir.mkdir(exist_ok=True)
                
                # Run evaluation on the temp directory
                main_by_model(
                    num_samples=NUM_SAMPLES, 
                    model=eval_model, 
                    data_dir=str(temp_dir),  # Use the temp directory
                    baseline_type=baseline_type, 
                    use_same_model=False,
                    results_dir=str(category_results_dir),
                    transcripts_dir=str(category_transcripts_dir),
                    model_prefix=f"{category_name}_gpt_oss_120b"
                )
                
                print(f"✅ {baseline_type} completed successfully for {category_name}")
            except Exception as e:
                print(f"❌ {baseline_type} failed for {category_name}: {e}")
                success = False
        
        if success:
            print(f"✅ All evaluations completed successfully for {category_name}")
            
            # Update S3 files with baseline results if enabled
            if UPDATE_S3:
                print(f"\n🔄 Updating S3 files with baseline results for {category_name}...")
                try:
                    # Get the baseline results directory for this category
                    baseline_dir = Path(__file__).parent
                    results_dir = baseline_dir / "results"
                    
                    # Run post-processing to update S3 files
                    s3_results = post_process_results(str(results_dir))
                    
                    # Count successful S3 updates for this category
                    s3_updates = 0
                    for baseline_type, tasks in s3_results.items():
                        for task_name, samples in tasks.items():
                            for sample in samples:
                                if sample.get('baseline_written', False):
                                    s3_updates += 1
                    
                    print(f"✅ Successfully updated {s3_updates} S3 files with baseline results for {category_name}")
                    
                except Exception as e:
                    print(f"⚠️ Warning: Failed to update S3 files for {category_name}: {e}")
                    print("   Baseline results are still saved locally")
            
            return True
        else:
            print(f"❌ Some evaluations failed for {category_name}")
            return False
            
    except Exception as e:
        print(f"❌ Error running evaluation for {category_name}: {e}")
        return False
    
    finally:
        os.chdir(original_cwd)
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
            print(f"🧹 Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            print(f"⚠️ Could not clean up temporary directory {temp_dir}: {e}")

def get_category_directories(organized_dir: Path) -> list:
    """Get list of category directories from the OSS 120B data directory."""
    category_dirs = []
    for item in organized_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Check if it has train.jsonl and val.jsonl files
            train_file = item / "train.jsonl"
            val_file = item / "val.jsonl"
            if train_file.exists() and val_file.exists():
                category_dirs.append(item.name)
    return sorted(category_dirs)

def main():
    """Main function to run evaluation on all OSS 120B categories."""
    organized_dir = Path(ORGANIZED_DIR)
    
    if not organized_dir.exists():
        print(f"❌ Organized directory not found: {organized_dir}")
        return
    
    # Get category directories from the OSS 120B data directory
    category_dirs = get_category_directories(organized_dir)
    if not category_dirs:
        print(f"❌ No category directories found in {organized_dir}")
        return
    
    print(f"📋 Found category directories: {category_dirs}")
    
    print(f"\n⚙️ Configuration:")
    print(f"  ORGANIZED_DIR: {ORGANIZED_DIR}")
    print(f"  NUM_SAMPLES: {NUM_SAMPLES}")
    print(f"  BASELINE_TYPES: {BASELINE_TYPES}")
    print(f"  EVALUATION_MODEL: {EVALUATION_MODEL}")
    print(f"  UPDATE_S3: {UPDATE_S3}")
    
    # Run evaluation for each category
    successful_runs = 0
    total_runs = len(category_dirs)
    
    for category_name in category_dirs:
        if run_evaluation_for_category(category_name, EVALUATION_MODEL, organized_dir):
            successful_runs += 1
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"📊 EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total categories: {total_runs}")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {total_runs - successful_runs}")
    
    if successful_runs == total_runs:
        print(f"🎉 All evaluations completed successfully!")
    else:
        print(f"⚠️ Some evaluations failed. Check the logs above.")
    
    # Show results location
    print(f"\n📁 Results are organized by baseline type and category:")
    baseline_dir = Path(__file__).parent
    print(f"  Results: {baseline_dir}/results/[baseline_type]/[category]/")
    print(f"  Transcripts: {baseline_dir}/transcripts/[baseline_type]/[category]/")
    
    # Show S3 update status
    if UPDATE_S3:
        print(f"\n☁️ S3 Update Status:")
        print(f"  S3 updates were enabled and processed after each successful evaluation")
        print(f"  Baseline results have been written to corresponding S3 files")
    else:
        print(f"\n☁️ S3 Update Status:")
        print(f"  S3 updates were disabled (set UPDATE_S3=True to enable)")
        print(f"  To update S3 files manually, run: post_process_results('{baseline_dir}/results')")
    
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
