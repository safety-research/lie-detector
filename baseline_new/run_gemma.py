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

# Set the log directory to be within baseline_new instead of .data
current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["INSPECT_LOG_DIR"] = os.path.join(current_dir, "logs")

# Add the parent directory to Python path to ensure baseline module can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Configuration
ORGANIZED_DIR = "/Users/dipikakhullar/Desktop/lie-detector/.data/google"
NUM_SAMPLES = 10 # Set to a number to limit samples, or None for all
BASELINE_TYPES = ["llama_chat", "base_transcript", "rowans_escaped_transcript"]
EVALUATION_MODES = ["folds_colors"]

def load_model_mapping(organized_dir: Path) -> dict:
    """Load the model mapping from the JSON file."""
    mapping_file = organized_dir / "model_mapping.json"
    
    if not mapping_file.exists():
        raise FileNotFoundError(f"Model mapping file not found: {mapping_file}")
    
    with open(mapping_file, 'r') as f:
        mapping_data = json.load(f)
    
    return mapping_data["mapping"]

def get_model_directories(organized_dir: Path) -> list:
    """Get list of model directories from the google data directory."""
    model_dirs = []
    for item in organized_dir.iterdir():
        if item.is_dir() and item.name.startswith('gemma_3_'):
            model_dirs.append(item.name)
    return sorted(model_dirs)

def get_evaluation_datasets(model_dir: Path) -> dict:
    """Get available evaluation datasets for a model directory."""
    datasets = {}
    
    # Check for category-based splits (like ascii, games, etc.)
    category_datasets = {}
    for category_dir in model_dir.iterdir():
        if category_dir.is_dir():
            train_file = category_dir / "train.jsonl"
            val_file = category_dir / "val.jsonl"
            if train_file.exists() and val_file.exists():
                category_datasets[category_dir.name] = {
                    "train": str(train_file),
                    "test": str(val_file)  # Using val as test for compatibility
                }
    if category_datasets:
        datasets["categories"] = category_datasets
    
    return datasets





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
    
    # Get available evaluation datasets
    datasets = get_evaluation_datasets(model_dir)
    if not datasets:
        print(f"❌ No evaluation datasets found in {model_dir}")
        return False
    
    print(f"📊 Found datasets: {list(datasets.keys())}")
    
    # Change to the model directory for evaluation
    original_cwd = os.getcwd()
    os.chdir(model_dir)
    
    try:
        # Import evaluation functions
        from evaluate import main_by_model
        
        print(f"📁 Running evaluation from: {model_dir}")
        
        # Set up top-level results directories
        organized_dir = Path(ORGANIZED_DIR)
        # Save results in baseline/results directory
        baseline_dir = Path(__file__).parent
        top_level_results = baseline_dir / "results"
        top_level_transcripts = baseline_dir / "transcripts"
        
        # Create directories if they don't exist
        top_level_results.mkdir(exist_ok=True)
        top_level_transcripts.mkdir(exist_ok=True)
        
        # Run evaluation for each baseline type and dataset
        success = True
        for baseline_type in BASELINE_TYPES:
            print(f"\n{'='*40}\nRunning baseline: {baseline_type}\n{'='*40}")
            
            try:
                # Create baseline-specific directories
                baseline_results_dir = top_level_results / baseline_type
                baseline_transcripts_dir = top_level_transcripts / baseline_type
                baseline_results_dir.mkdir(exist_ok=True)
                baseline_transcripts_dir.mkdir(exist_ok=True)
                
                # Run evaluation for each dataset
                for dataset_type, dataset_info in datasets.items():
                    print(f"\n--- Processing dataset: {dataset_type} ---")
                    
                    if dataset_type == "categories":
                        # Run on each category (train only)
                        for category_name, category_info in dataset_info.items():
                            print(f"  Processing category: {category_name} (train only)")
                            success &= run_evaluation_on_split(
                                model_dir, eval_model, baseline_type, 
                                category_info["train"], f"{category_name}_train", 
                                baseline_results_dir, baseline_transcripts_dir, model_folder
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

def run_evaluation_on_split(model_dir: Path, eval_model: str, baseline_type: str, 
                           data_file: str, split_name: str, 
                           results_dir: Path, transcripts_dir: Path, model_folder: str) -> bool:
    """Run evaluation on a specific data split."""
    try:
        # Import evaluation functions
        from evaluate import main_by_model
        
        # Create split-specific directories
        split_results_dir = results_dir / split_name
        split_transcripts_dir = transcripts_dir / split_name
        split_results_dir.mkdir(exist_ok=True)
        split_transcripts_dir.mkdir(exist_ok=True)
        
        # Temporarily change working directory to model directory for data access
        original_cwd = os.getcwd()
        os.chdir(model_dir)
        
        try:
            # Run the evaluation using the imported function with custom output directories
            # data_file is a file path, but main_by_model expects a directory path
            # So we need to pass the directory containing the file
            data_dir = Path(data_file).parent
            main_by_model(
                num_samples=NUM_SAMPLES, 
                model=eval_model, 
                data_dir=str(data_dir),  # Use the directory containing the data file
                baseline_type=baseline_type, 
                use_same_model=False,
                results_dir=str(split_results_dir),
                transcripts_dir=str(split_transcripts_dir),
                model_prefix=f"{model_folder}_{split_name}"
            )
            
            print(f"    ✅ {split_name} completed successfully")
            return True
        finally:
            os.chdir(original_cwd)
            
    except Exception as e:
        print(f"    ❌ {split_name} failed: {e}")
        return False

def aggregate_results(results_dir: Path, model_folder: str, dataset_type: str):
    """Aggregate results from train and test splits."""
    try:
        train_results = results_dir / "train" / f"{model_folder}_train_results.json"
        test_results = results_dir / "test" / f"{model_folder}_test_results.json"
        
        if train_results.exists() and test_results.exists():
            # Load and combine results
            with open(train_results, 'r') as f:
                train_data = json.load(f)
            with open(test_results, 'r') as f:
                test_data = json.load(f)
            
            # Create aggregated results
            aggregated_results = {
                "model": model_folder,
                "dataset_type": dataset_type,
                "train": train_data,
                "test": test_data,
                "aggregated": {
                    "train_accuracy": train_data.get("accuracy", 0),
                    "test_accuracy": test_data.get("accuracy", 0),
                    "train_f1": train_data.get("f1", 0),
                    "test_f1": test_data.get("f1", 0)
                }
            }
            
            # Save aggregated results
            aggregated_file = results_dir / f"{model_folder}_{dataset_type}_aggregated.json"
            with open(aggregated_file, 'w') as f:
                json.dump(aggregated_results, f, indent=2)
            
            print(f"    📊 Aggregated results saved to: {aggregated_file}")
    except Exception as e:
        print(f"    ⚠️ Could not aggregate results: {e}")

def aggregate_fold_results(results_dir: Path, model_folder: str, fold_names):
    """Aggregate results from all folds."""
    try:
        fold_results = {}
        for fold_name in fold_names:
            train_file = results_dir / f"{fold_name}_train" / f"{model_folder}_{fold_name}_train_results.json"
            test_file = results_dir / f"{fold_name}_test" / f"{model_folder}_{fold_name}_test_results.json"
            
            if train_file.exists() and test_file.exists():
                with open(train_file, 'r') as f:
                    fold_results[f"{fold_name}_train"] = json.load(f)
                with open(test_file, 'r') as f:
                    fold_results[f"{fold_name}_test"] = json.load(f)
        
        if fold_results:
            # Calculate average metrics across folds
            test_accuracies = []
            test_f1s = []
            
            for fold_name in fold_names:
                test_key = f"{fold_name}_test"
                if test_key in fold_results:
                    test_accuracies.append(fold_results[test_key].get("accuracy", 0))
                    test_f1s.append(fold_results[test_key].get("f1", 0))
            
            aggregated_fold_results = {
                "model": model_folder,
                "dataset_type": "folds_colors_chat_format",
                "fold_results": fold_results,
                "aggregated": {
                    "mean_test_accuracy": sum(test_accuracies) / len(test_accuracies) if test_accuracies else 0,
                    "mean_test_f1": sum(test_f1s) / len(test_f1s) if test_f1s else 0,
                    "std_test_accuracy": 0,  # Could calculate standard deviation if needed
                    "std_test_f1": 0
                }
            }
            
            # Save aggregated fold results
            aggregated_file = results_dir / f"{model_folder}_folds_aggregated.json"
            with open(aggregated_file, 'w') as f:
                json.dump(aggregated_fold_results, f, indent=2)
            
            print(f"    📊 Fold results aggregated and saved to: {aggregated_file}")
    except Exception as e:
        print(f"    ⚠️ Could not aggregate fold results: {e}")

def main():
    """Main function to run evaluation on all organized model directories."""
    organized_dir = Path(ORGANIZED_DIR)
    
    if not organized_dir.exists():
        print(f"❌ Organized directory not found: {organized_dir}")
        return
    
    # Get model directories from the organized balanced training directory
    model_dirs = get_model_directories(organized_dir)
    if not model_dirs:
        print(f"❌ No model directories found in {organized_dir}")
        return
    
    print(f"📋 Found model directories: {model_dirs}")
    
    # Create a simple model mapping (assuming the model name is derived from the directory name)
    model_mapping = {}
    for model_dir in model_dirs:
        # Extract model name from directory (e.g., "gemma_3_4b_it" -> "openrouter/google/gemma-3-4b-it")
        if "gemma_3_4b_it" in model_dir:
            model_mapping[model_dir] = "openrouter/google/gemma-3-4b-it"
        elif "gemma_3_12b_it" in model_dir:
            model_mapping[model_dir] = "openrouter/google/gemma-3-12b-it"
        elif "gemma_3_27b_it" in model_dir:
            model_mapping[model_dir] = "openrouter/google/gemma-3-27b-it"
        else:
            model_mapping[model_dir] = "unknown"
    
    print(f"📋 Model mapping created:")
    for folder, model in model_mapping.items():
        print(f"  {folder} -> {model}")
    
    print(f"\n⚙️ Configuration:")
    print(f"  NUM_SAMPLES: {NUM_SAMPLES}")
    print(f"  BASELINE_TYPES: {BASELINE_TYPES}")
    print(f"  EVALUATION MODES: {EVALUATION_MODES}")
    print(f"  EVALUATION MODE: Use model from model mapping for each directory")
    
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
    baseline_dir = Path(__file__).parent
    print(f"  Results: {baseline_dir}/results/[baseline_type]/")
    print(f"  Transcripts: {baseline_dir}/transcripts/[baseline_type]/")
    
    print(f"{'='*80}")

if __name__ == "__main__":
    main() 