#!/usr/bin/env python3
"""
Evaluation script for baseline_new directory.
Handles all baseline types including the new gt_labelling_rowan_modified.
"""

from inspect_ai import eval
from dotenv import load_dotenv
import json
import os
import sys
from pathlib import Path
from datetime import datetime

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles non-serializable objects."""
    def default(self, obj):
        if hasattr(obj, '__dict__'):
            # For objects with __dict__, try to serialize their attributes
            try:
                return obj.__dict__
            except:
                return str(obj)
        elif hasattr(obj, '__class__'):
            # For other objects, convert to string
            return str(obj)
        return super().default(obj)

# Set up OpenRouter authentication
if "OPENROUTER_API_KEY" in os.environ:
    os.environ["OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"]
    os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
    print("✅ OpenRouter authentication configured")
else:
    print("⚠️ Warning: OPENROUTER_API_KEY not found in environment variables")

# Add the parent directory to Python path to ensure baseline module can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import from current directory
from task import baseline_task, escaped_transcript_task, llama_chat_task, llama_chat_reasoning_task, \
    base_transcript_reasoning_task, rowans_escaped_transcript_task, gt_labelling_rowan_modified_task
from task import baseline_task_by_model, escaped_transcript_task_by_model, llama_chat_task_by_model, \
    llama_chat_reasoning_task_by_model, base_transcript_reasoning_task_by_model, \
    rowans_escaped_transcript_task_by_model, gt_labelling_rowan_modified_task_by_model

load_dotenv()


def extract_output_from_sample(sample):
    """Extract the model output from a sample, trying multiple sources."""
    # First try the standard output.completion
    if hasattr(sample, 'output') and hasattr(sample.output, 'completion') and sample.output.completion:
        return sample.output.completion
    
    # Try the output object directly
    if hasattr(sample, 'output') and sample.output:
        return str(sample.output)
    
    # Try to extract from metadata messages (last assistant message)
    if hasattr(sample, 'metadata') and sample.metadata:
        messages = sample.metadata.get('messages', [])
        # Find the last assistant message
        for msg in reversed(messages):
            if msg.get('role') == 'assistant':
                return msg.get('content', '')
    
    return None


def main_by_model(num_samples=None, model=None, data_dir="processed_data", baseline_type="base_transcript",
                  use_same_model=True, results_dir=None, transcripts_dir=None, model_prefix=None):
    """Run baseline evaluation separately for each model."""
    print("\n" + "=" * 80)
    print(f"=== RUNNING {baseline_type.upper()} EVALUATION BY MODEL ===")

    # Configuration
    default_model = model or "openrouter/meta-llama/llama-3.1-8b-instruct"
    log_dir = f"logs/{baseline_type}"
    limit = num_samples

    print(f"Running {baseline_type} evaluation by model...")
    if use_same_model:
        print(f"Using same model for evaluation as the one that generated the lies")
    else:
        print(f"Using fixed evaluation model: {default_model}")
    print(f"Samples per model: {'All available' if limit is None else limit}")

    # Choose task based on baseline type
    if baseline_type == "escaped_transcript":
        model_tasks = escaped_transcript_task_by_model(data_dir=data_dir, limit=limit)
    elif baseline_type == "llama_chat":
        model_tasks = llama_chat_task_by_model(data_dir=data_dir, limit=limit)
    elif baseline_type == "llama_chat_reasoning":
        model_tasks = llama_chat_reasoning_task_by_model(data_dir=data_dir, limit=limit)
    elif baseline_type == "base_transcript_reasoning":
        model_tasks = base_transcript_reasoning_task_by_model(data_dir=data_dir, limit=limit)
    elif baseline_type == "rowans_escaped_transcript":
        model_tasks = rowans_escaped_transcript_task_by_model(data_dir=data_dir, limit=limit)
    elif baseline_type == "gt_labelling_rowan_modified":
        model_tasks = gt_labelling_rowan_modified_task_by_model(data_dir=data_dir, limit=limit)
    else:  # base_transcript or fallback
        model_tasks = baseline_task_by_model(data_dir=data_dir, limit=limit)

    print(f"\nFound {len(model_tasks)} models to evaluate:")
    for model_key in model_tasks.keys():
        print(f"  - {model_key}")

    # Create output directories
    if results_dir is None:
        results_dir = f"results/{baseline_type}"
    if transcripts_dir is None:
        transcripts_dir = f"transcripts/{baseline_type}"

    # Create directories
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    Path(transcripts_dir).mkdir(parents=True, exist_ok=True)

    # Run evaluation for each model
    successful_models = 0
    total_models = len(model_tasks)

    for model_name, task in model_tasks.items():
        print(f"\n{'='*60}")
        print(f"🎯 EVALUATING MODEL: {model_name}")
        print(f"{'='*60}")

        try:
            # Determine which model to use for evaluation
            if use_same_model:
                # Map the data model to the evaluation model
                eval_model = map_data_model_to_eval_model(model_name)
            else:
                eval_model = default_model

            print(f"Using evaluation model: {eval_model}")

            # Run evaluation
            log = eval(task, model=eval_model, log_dir=log_dir)
            results = log[0].samples if log and log[0].samples else []

            print(f"Processed {len(results)} samples for {model_name}")

            # Save results
            if results:
                # Extract category from model_prefix (e.g., "cot-unfaithfulness_openai_gpt-oss-120b" -> "cot-unfaithfulness")
                category = model_prefix.split('_')[0] if model_prefix and '_' in model_prefix else "unknown"
                
                # Clean model_name to remove any slashes for filename safety
                clean_model_name = model_name.replace('/', '_').replace('\\', '_')
                
                # Create category-specific subdirectories
                category_results_dir = Path(results_dir) / category
                category_transcripts_dir = Path(transcripts_dir) / category
                category_results_dir.mkdir(parents=True, exist_ok=True)
                category_transcripts_dir.mkdir(parents=True, exist_ok=True)

                # Generate timestamp for filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Calculate aggregated confusion matrix metrics
                aggregated_metrics = {}
                for scorer_name in ['escaped_transcript_scorer', 'truth_scorer', 'llama_chat_scorer', 'llama_chat_reasoning_scorer', 'base_transcript_reasoning_scorer']:
                    if hasattr(results[0], 'scores') and results[0].scores and scorer_name in results[0].scores:
                        total_tp = sum(score.metadata.get('tp', 0) for sample in results for score in [sample.scores[scorer_name]] if hasattr(sample, 'scores') and sample.scores and scorer_name in sample.scores)
                        total_tn = sum(score.metadata.get('tn', 0) for sample in results for score in [sample.scores[scorer_name]] if hasattr(sample, 'scores') and sample.scores and scorer_name in sample.scores)
                        total_fp = sum(score.metadata.get('fp', 0) for sample in results for score in [sample.scores[scorer_name]] if hasattr(sample, 'scores') and sample.scores and scorer_name in sample.scores)
                        total_fn = sum(score.metadata.get('fn', 0) for sample in results for score in [sample.scores[scorer_name]] if hasattr(sample, 'scores') and sample.scores and scorer_name in sample.scores)
                        
                        # Calculate aggregated metrics
                        total_samples = len(results)
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

                # Save results to JSON (simplified format)
                results_file = category_results_dir / f"{category}_{clean_model_name}_results_{timestamp}.json"
                with open(results_file, 'w') as f:
                    # Create simplified samples with only the requested fields
                    simplified_samples = []
                    for sample in results:
                        # Extract old_gt from target
                        old_gt = sample.target if sample.target else "unknown"
                        
                        # Extract baseline prediction from output
                        baseline_prediction = extract_output_from_sample(sample)
                        if not baseline_prediction:
                            baseline_prediction = "unknown"
                        
                        # Extract trace from metadata (messages array)
                        trace = ''
                        if sample.metadata and 'messages' in sample.metadata:
                            # Convert messages to a readable trace format
                            messages = sample.metadata['messages']
                            trace_parts = []
                            for i, msg in enumerate(messages):
                                role = msg.get('role', 'unknown')
                                content = msg.get('content', '')
                                trace_parts.append(f"Turn {i+1} ({role}): {content[:200]}{'...' if len(content) > 200 else ''}")
                            trace = '\n'.join(trace_parts)
                        
                        # Extract sample_id from metadata (prioritize s3_metadata)
                        sample_id = ''
                        if sample.metadata:
                            if 's3_metadata' in sample.metadata and 'sample_id' in sample.metadata['s3_metadata']:
                                sample_id = str(sample.metadata['s3_metadata']['sample_id'])
                            elif 'meta' in sample.metadata and 'sample_id' in sample.metadata['meta']:
                                sample_id = str(sample.metadata['meta']['sample_id'])
                            elif 'sample_id' in sample.metadata:
                                sample_id = str(sample.metadata['sample_id'])
                        
                        # Extract task_name from metadata (prioritize s3_metadata)
                        task_name = ''
                        if sample.metadata:
                            if 's3_metadata' in sample.metadata and 'task' in sample.metadata['s3_metadata']:
                                task_name = str(sample.metadata['s3_metadata']['task'])
                            elif 'meta' in sample.metadata and 'task' in sample.metadata['meta']:
                                task_name = str(sample.metadata['meta']['task'])
                            elif 'task' in sample.metadata:
                                task_name = str(sample.metadata['task'])
                        
                        # Extract original_dataset_sample_id (task_id from s3_metadata)
                        original_dataset_sample_id = ''
                        if sample.metadata:
                            if 's3_metadata' in sample.metadata and 'task_id' in sample.metadata['s3_metadata']:
                                original_dataset_sample_id = str(sample.metadata['s3_metadata']['task_id'])
                            elif 'meta' in sample.metadata and 'task_id' in sample.metadata['meta']:
                                original_dataset_sample_id = str(sample.metadata['meta']['task_id'])
                            elif 'task_id' in sample.metadata:
                                original_dataset_sample_id = str(sample.metadata['task_id'])
                        
                        simplified_samples.append({
                            'old_gt': old_gt,
                            'baseline_prediction': baseline_prediction,
                            'trace': trace,
                            'sample_id': sample_id,
                            'original_dataset_sample_id': original_dataset_sample_id,
                            'task_name': task_name
                        })
                    
                    json.dump({
                        'aggregated_metrics': aggregated_metrics,
                        'samples': simplified_samples
                    }, f, indent=2, cls=CustomJSONEncoder)

                # Save transcripts
                transcripts_file = category_transcripts_dir / f"{category}_{clean_model_name}_transcripts_{timestamp}.txt"
                with open(transcripts_file, 'w') as f:
                    for i, sample in enumerate(results):
                        f.write(f"=== SAMPLE {i+1} ===\n")
                        f.write(f"Input: {sample.input}\n")
                        f.write(f"Target: {sample.target}\n")
                        output = extract_output_from_sample(sample)
                        if output:
                            f.write(f"Output: {output}\n")
                        f.write(f"Metadata: {sample.metadata}\n")
                        if hasattr(sample, 'scores') and sample.scores:
                            f.write(f"Scores: {sample.scores}\n")
                        f.write("\n" + "="*80 + "\n")

                print(f"✅ Results saved to: {results_file}")
                print(f"✅ Transcripts saved to: {transcripts_file}")
                
                # Note: S3 writing for GT labelling is handled by post_process_results() after evaluation completes
                
                successful_models += 1
            else:
                print(f"⚠️ No results generated for {model_name}")

        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    print(f"\n{'='*80}")
    print(f"📊 EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total models: {total_models}")
    print(f"Successful models: {successful_models}")
    print(f"Failed models: {total_models - successful_models}")
    print(f"Baseline type: {baseline_type}")
    print(f"Results directory: {results_dir}")
    print(f"Transcripts directory: {transcripts_dir}")

    if successful_models == total_models:
        print(f"🎉 All models evaluated successfully!")
    else:
        print(f"⚠️ Some models failed. Check the logs above.")


def map_data_model_to_eval_model(data_model: str) -> str:
    """
    Map the model name from the data to the actual model name for evaluation.
    This ensures we use the same model for evaluation as the one that generated the lies.
    """
    # Normalize the data model name (remove provider prefixes if present)
    if data_model.startswith('openrouter/'):
        data_model = data_model[11:]  # Remove 'openrouter/'
    elif data_model.startswith('anthropic/'):
        data_model = data_model[10:]  # Remove 'anthropic/'
    elif data_model.startswith('openai/'):
        data_model = data_model[7:]  # Remove 'openai/'

    # Map to the full model name for evaluation
    model_mapping = {
        'gpt-oss-120b': 'openrouter/openai/gpt-oss-120b',
        'gpt-oss-20b': 'openrouter/openai/gpt-oss-20b',
        'claude-opus-4': 'openrouter/anthropic/claude-opus-4',
        'claude-3-5-sonnet-20241022': 'openrouter/anthropic/claude-3-5-sonnet-20241022',
        'llama-3.1-405b-instruct': 'openrouter/meta-llama/llama-3.1-405b-instruct',
        'gemini-2.0-flash-exp': 'openrouter/google/gemini-2.0-flash-exp',
    }

    # Return mapped model or add openrouter prefix if not found
    if data_model in model_mapping:
        return model_mapping[data_model]
    else:
        return f"openrouter/{data_model}"


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run baseline evaluation")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to process")
    parser.add_argument("--model", type=str, default=None, help="Model to use for evaluation")
    parser.add_argument("--data_dir", type=str, default="processed_data", help="Data directory")
    parser.add_argument("--baseline_type", type=str, default="base_transcript", help="Baseline type")
    parser.add_argument("--use_same_model", action="store_true", help="Use same model as in data")
    parser.add_argument("--results_dir", type=str, default=None, help="Results directory")
    parser.add_argument("--transcripts_dir", type=str, default=None, help="Transcripts directory")
    parser.add_argument("--model_prefix", type=str, default=None, help="Model prefix for output files")
    
    args = parser.parse_args()
    
    main_by_model(
        num_samples=args.num_samples,
        model=args.model,
        data_dir=args.data_dir,
        baseline_type=args.baseline_type,
        use_same_model=args.use_same_model,
        results_dir=args.results_dir,
        transcripts_dir=args.transcripts_dir,
        model_prefix=args.model_prefix
    )
