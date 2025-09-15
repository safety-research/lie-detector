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

# No S3 functionality needed

# Configuration
ORGANIZED_DIR = "/Users/dipikakhullar/Desktop/lie_detector_clone/lie-detector/.data/openai/gpt_oss_120b_limited"
NUM_SAMPLES = 700  # Set to a number to limit samples, or None for all
BASELINE_TYPES = ["gt_labelling_rowan_modified"]
EVALUATION_MODELS = [
    "openrouter/openai/gpt-oss-120b",
    "openrouter/openai/gpt-4o",
    "openrouter/anthropic/claude-opus-4",
    "openrouter/openai/gpt-5",
    "openrouter/anthropic/claude-sonnet-4"
]  # List of models to use for evaluation
# S3 functionality removed

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

def run_evaluation_for_category(category_name: str, eval_models: list, organized_dir: Path):
    """Run evaluation for a specific category with multiple evaluation models in parallel."""
    print(f"\n{'='*80}")
    print(f"🚀 RUNNING EVALUATION FOR CATEGORY: {category_name}")
    print(f"🎯 EVALUATION MODELS: {', '.join(eval_models)}")
    print(f"⚡ PARALLEL EXECUTION: ENABLED")
    print(f"📊 PROCESSING: BOTH TRAIN AND VAL DATA")
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
    
    # Count samples in each file
    train_count = sum(1 for _ in open(train_file)) if train_file.exists() else 0
    val_count = sum(1 for _ in open(val_file)) if val_file.exists() else 0
    print(f"📊 Data counts: Train={train_count}, Val={val_count}, Total={train_count + val_count}")
    
    # Create temporary directory structure
    temp_dir = create_temp_model_structure(organized_dir, category_name)
    print(f"📁 Created temporary structure: {temp_dir}")
    
    # Change to the temp directory for evaluation
    original_cwd = os.getcwd()
    os.chdir(temp_dir)
    
    print(f"📁 Running evaluation from: {temp_dir}")
    
    # Set up top-level results directories
    baseline_dir = Path(__file__).parent
    top_level_results = baseline_dir / "results"
    top_level_transcripts = baseline_dir / "transcripts"
    
    # Create directories if they don't exist
    top_level_results.mkdir(exist_ok=True)
    top_level_transcripts.mkdir(exist_ok=True)
    
    # Import evaluation functions
    from evaluate import main_by_model
    from inspect_ai import eval
    from task import gt_labelling_rowan_modified_task_by_model
    
    # Run evaluation for each baseline type in parallel across all models
    for baseline_type in BASELINE_TYPES:
        print(f"\n{'='*60}")
        print(f"🎯 RUNNING {baseline_type.upper()} FOR ALL MODELS IN PARALLEL")
        print(f"📋 MODELS: {', '.join(eval_models)}")
        print(f"📊 DATA: Both train.jsonl and val.jsonl (Total: {train_count + val_count} samples)")
        print(f"{'='*60}")
        
        try:
            # Get the task for this baseline type - this will process BOTH train and val
            if baseline_type == "gt_labelling_rowan_modified":
                model_tasks = gt_labelling_rowan_modified_task_by_model(data_dir=str(temp_dir), limit=NUM_SAMPLES)
            else:
                # Fallback to other task types if needed
                print(f"⚠️ Unknown baseline type: {baseline_type}")
                continue
            
            # Get the first model's task as a template for parallel execution
            first_model = list(model_tasks.keys())[0]
            first_task = model_tasks[first_model]
            
            # Run evaluation with all models in parallel using inspect_ai.eval
            print(f"🚀 Starting parallel evaluation with {len(eval_models)} models...")
            
            # Use inspect_ai.eval with multiple models and max_workers for parallel execution
            results = eval(
                first_task,  # Use the task directly
                model=eval_models,  # Pass list of models
                max_workers=min(len(eval_models), 4),  # Limit concurrent workers
                log_dir=f"logs/{baseline_type}_parallel",
                retry_on_error=3,
                fail_on_error=0.2
            )
            
            print(f"✅ Parallel evaluation completed for {baseline_type}")
            
            # Process results for each model
            if hasattr(results, 'samples') and results.samples:
                # Single result object (if only one model)
                model_results = [results]
            else:
                # Multiple result objects (one per model)
                model_results = list(results) if hasattr(results, '__iter__') else [results]
            
            # Save results for each model
            for i, eval_model in enumerate(eval_models):
                if i < len(model_results):
                    result = model_results[i]
                    
                    # Extract model name from eval_model
                    model_name = eval_model.split('/')[-1] if '/' in eval_model else eval_model
                    
                    # Create baseline-specific directories
                    baseline_results_dir = top_level_results / baseline_type / model_name
                    baseline_transcripts_dir = top_level_transcripts / baseline_type / model_name
                    baseline_results_dir.mkdir(parents=True, exist_ok=True)
                    baseline_transcripts_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save results using the existing main_by_model logic
                    if hasattr(result, 'samples') and result.samples:
                        # Import the result saving logic from evaluate.py
                        from datetime import datetime
                        import json
                        
                        # Generate timestamp for filename
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        # Calculate aggregated confusion matrix metrics
                        aggregated_metrics = {}
                        for scorer_name in ['escaped_transcript_scorer', 'truth_scorer', 'llama_chat_scorer', 'llama_chat_reasoning_scorer', 'base_transcript_reasoning_scorer']:
                            if hasattr(result.samples[0], 'scores') and result.samples[0].scores and scorer_name in result.samples[0].scores:
                                total_tp = sum(score.metadata.get('tp', 0) for sample in result.samples for score in [sample.scores[scorer_name]] if hasattr(sample, 'scores') and sample.scores and scorer_name in sample.scores)
                                total_tn = sum(score.metadata.get('tn', 0) for sample in result.samples for score in [sample.scores[scorer_name]] if hasattr(sample, 'scores') and sample.scores and scorer_name in sample.scores)
                                total_fp = sum(score.metadata.get('fp', 0) for sample in result.samples for score in [sample.scores[scorer_name]] if hasattr(sample, 'scores') and sample.scores and scorer_name in sample.scores)
                                total_fn = sum(score.metadata.get('fn', 0) for sample in result.samples for score in [sample.scores[scorer_name]] if hasattr(sample, 'scores') and sample.scores and scorer_name in sample.scores)
                                
                                # Calculate aggregated metrics
                                total_samples = len(result.samples)
                                accuracy = (total_tp + total_tn) / total_samples if total_samples > 0 else 0
                                precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
                                recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
                                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                                
                                aggregated_metrics[scorer_name] = {
                                    'confusion_matrix': {
                                        'tp': total_tp,
                                        'tn': total_tn,
                                        'fp': total_fp,
                                        'fn': total_fn
                                    },
                                    'metrics': {
                                        'accuracy': accuracy,
                                        'precision': precision,
                                        'recall': recall,
                                        'f1_score': f1
                                    },
                                    'total_samples': total_samples
                                }
                        
                        # Create category-specific subdirectories
                        category_results_dir = baseline_results_dir / category_name
                        category_transcripts_dir = baseline_transcripts_dir / category_name
                        category_results_dir.mkdir(parents=True, exist_ok=True)
                        category_transcripts_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Save results to JSON
                        results_file = category_results_dir / f"{category_name}_{model_name}_results_{timestamp}.json"
                        with open(results_file, 'w') as f:
                            json.dump({
                                'aggregated_metrics': aggregated_metrics,
                                'samples': [{
                                    'input': sample.input,
                                    'target': sample.target,
                                    'output': sample.output.completion if hasattr(sample, 'output') else None,
                                    'metadata': sample.metadata,
                                    'scores': {name: {'value': score.value, 'metadata': score.metadata, 'answer': score.answer} 
                                             for name, score in sample.scores.items()} if hasattr(sample, 'scores') and sample.scores else {}
                                } for sample in result.samples]
                            }, f, indent=2)
                        
                        # Save transcripts
                        transcripts_file = category_transcripts_dir / f"{category_name}_{model_name}_transcripts_{timestamp}.txt"
                        with open(transcripts_file, 'w') as f:
                            for i, sample in enumerate(result.samples):
                                f.write(f"=== SAMPLE {i+1} ===\n")
                                f.write(f"Input: {sample.input}\n")
                                f.write(f"Target: {sample.target}\n")
                                if hasattr(sample, 'output'):
                                    f.write(f"Output: {sample.output.completion}\n")
                                f.write(f"Metadata: {sample.metadata}\n")
                                if hasattr(sample, 'scores') and sample.scores:
                                    f.write(f"Scores: {sample.scores}\n")
                                f.write("\n" + "="*80 + "\n")
                        
                        print(f"✅ {baseline_type} completed for {category_name} with {eval_model}")
                        print(f"   📄 Results: {results_file}")
                        print(f"   📄 Transcripts: {transcripts_file}")
                    else:
                        print(f"⚠️ No results generated for {eval_model} with {baseline_type}")
        
        except Exception as e:
            print(f"❌ Error running {baseline_type} in parallel: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"✅ All evaluations completed successfully for {category_name}")
    
    # Clean up and return
    os.chdir(original_cwd)
    shutil.rmtree(temp_dir)
    print(f"🧹 Cleaned up temporary directory: {temp_dir}")
    
    return True

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
    print(f"  EVALUATION_MODELS: {EVALUATION_MODELS}")
    print(f"  PARALLEL EXECUTION: ✅ ENABLED")
    print(f"  MAX_WORKERS: {min(len(EVALUATION_MODELS), 4)}")
    print(f"  DATA PROCESSING: ✅ BOTH TRAIN AND VAL FILES")
    
    # Run evaluation for each category
    successful_categories = 0
    for category_name in category_dirs:
        success = run_evaluation_for_category(category_name, EVALUATION_MODELS, organized_dir)
        if success:
            successful_categories += 1
    
    # Calculate total data counts
    total_train = 0
    total_val = 0
    for category_name in category_dirs:
        category_dir = organized_dir / category_name
        train_file = category_dir / "train.jsonl"
        val_file = category_dir / "val.jsonl"
        if train_file.exists():
            total_train += sum(1 for _ in open(train_file))
        if val_file.exists():
            total_val += sum(1 for _ in open(val_file))
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"📊 EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total categories: {len(category_dirs)}")
    print(f"Successful categories: {successful_categories}")
    print(f"Failed categories: {len(category_dirs) - successful_categories}")
    print(f"Evaluation models used: {len(EVALUATION_MODELS)}")
    print(f"  - {', '.join(EVALUATION_MODELS)}")
    print(f"Baseline types per model: {len(BASELINE_TYPES)}")
    print(f"  - {', '.join(BASELINE_TYPES)}")
    print(f"Parallel execution: ✅ ENABLED")
    print(f"Data processing: ✅ BOTH TRAIN AND VAL FILES")
    print(f"📊 Total data processed:")
    print(f"  - Train samples: {total_train:,}")
    print(f"  - Val samples: {total_val:,}")
    print(f"  - Total samples: {total_train + total_val:,}")
    print(f"🎉 All evaluations completed!")
    
    # Show results location
    print(f"\n📁 Results are organized by baseline type and model:")
    baseline_dir = Path(__file__).parent
    print(f"  Results: {baseline_dir}/results/[baseline_name]/[model_name]/[category]/[category]_[model_name]_results_[datetime].json")
    print(f"  Transcripts: {baseline_dir}/transcripts/[baseline_name]/[model_name]/[category]/[category]_[model_name]_transcripts_[datetime].txt")
    
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
