#!/usr/bin/env python3
"""
Run GT labelling evaluation on multiple judge models in parallel.
This script processes the gt_labelling_rowan_modified baseline type and writes
results back to S3 with the specified structure.
"""

import json
import os
import sys
import asyncio
import concurrent.futures
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import boto3
from botocore.exceptions import ClientError

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from task import gt_labelling_rowan_modified_task_by_model
from inspect_ai import eval
from inspect_ai.model import get_model


# Configuration
JUDGE_MODELS = [
    "openrouter/openai/gpt-4o",
    "openrouter/anthropic/claude-3-5-sonnet-20241022",
    "openrouter/meta-llama/llama-3.1-405b-instruct",
    "openrouter/google/gemini-2.0-flash-exp"
]

EXPERIMENT_NAME = "gt_labelling_experiment_1"  # Configurable experiment name
S3_BUCKET = "your-s3-bucket-name"  # Update with actual bucket name
S3_PREFIX = "lie-detector-results"  # Update with actual prefix
NUM_SAMPLES = 10  # Set to None for all samples
MAX_WORKERS = 4  # Number of parallel workers


def get_s3_client():
    """Get S3 client with credentials."""
    return boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
    )


def load_sample_from_s3(sample_id: str) -> Optional[Dict[str, Any]]:
    """Load a sample from S3 by sample_id."""
    try:
        s3_client = get_s3_client()
        key = f"{S3_PREFIX}/samples/{sample_id}.json"
        
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        content = response['Body'].read().decode('utf-8')
        return json.loads(content)
    except ClientError as e:
        print(f"❌ Error loading sample {sample_id} from S3: {e}")
        return None


def write_sample_to_s3(sample_id: str, sample_data: Dict[str, Any]) -> bool:
    """Write a sample back to S3 with updated ground_truth data."""
    try:
        s3_client = get_s3_client()
        key = f"{S3_PREFIX}/samples/{sample_id}.json"
        
        # Upload the updated sample data
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=json.dumps(sample_data, indent=2),
            ContentType='application/json'
        )
        return True
    except ClientError as e:
        print(f"❌ Error writing sample {sample_id} to S3: {e}")
        return False


def process_sample_with_model(sample: Any, model_name: str, experiment_name: str) -> Dict[str, Any]:
    """Process a single sample with a specific model."""
    try:
        # Get the model
        model = get_model(model_name)
        
        # Create a simple task for this sample
        from inspect_ai import Task, generate, MemoryDataset
        from scorer import escaped_transcript_scorer
        
        task = Task(
            dataset=MemoryDataset(samples=[sample]),
            solver=generate(),
            scorer=escaped_transcript_scorer()
        )
        
        # Run evaluation
        log = eval(task, model=model)
        result = log[0].samples[0] if log and log[0].samples else None
        
        if result and hasattr(result, 'scores') and 'escaped_transcript_scorer' in result.scores:
            score_data = result.scores['escaped_transcript_scorer']
            prediction = score_data.answer
            metadata = score_data.metadata
            
            return {
                "model_id": model_name,
                "prediction": prediction,
                "confidence": metadata.get("predicted_score"),
                "parseable": metadata.get("parseable", False),
                "raw_response": metadata.get("raw_response", ""),
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "model_id": model_name,
                "prediction": "ERROR",
                "confidence": None,
                "parseable": False,
                "raw_response": "",
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        print(f"❌ Error processing sample with {model_name}: {e}")
        return {
            "model_id": model_name,
            "prediction": "ERROR",
            "confidence": None,
            "parseable": False,
            "raw_response": str(e),
            "timestamp": datetime.now().isoformat()
        }


def process_sample_parallel(sample: Any, sample_id: str) -> bool:
    """Process a single sample with all judge models in parallel."""
    print(f"🔄 Processing sample {sample_id} with {len(JUDGE_MODELS)} judge models...")
    
    # Process with all models in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_model = {
            executor.submit(process_sample_with_model, sample, model, EXPERIMENT_NAME): model 
            for model in JUDGE_MODELS
        }
        
        model_results = {}
        for future in concurrent.futures.as_completed(future_to_model):
            model = future_to_model[future]
            try:
                result = future.result()
                model_results[model] = result
                print(f"✅ {model}: {result['prediction']} (confidence: {result['confidence']})")
            except Exception as e:
                print(f"❌ {model}: Error - {e}")
                model_results[model] = {
                    "model_id": model,
                    "prediction": "ERROR",
                    "confidence": None,
                    "parseable": False,
                    "raw_response": str(e),
                    "timestamp": datetime.now().isoformat()
                }
    
    # Load the sample from S3 to update it
    sample_data = load_sample_from_s3(sample_id)
    if not sample_data:
        print(f"❌ Could not load sample {sample_id} from S3")
        return False
    
    # Update the sample with ground_truth data
    if "ground_truth" not in sample_data:
        sample_data["ground_truth"] = {}
    
    if experiment_name not in sample_data["ground_truth"]:
        sample_data["ground_truth"][experiment_name] = {}
    
    # Add results from all models
    for model, result in model_results.items():
        sample_data["ground_truth"][experiment_name][model] = result
    
    # Write back to S3
    success = write_sample_to_s3(sample_id, sample_data)
    if success:
        print(f"✅ Successfully updated sample {sample_id} in S3")
    else:
        print(f"❌ Failed to update sample {sample_id} in S3")
    
    return success


def run_gt_labelling_evaluation(data_dir: str, num_samples: Optional[int] = None):
    """Run GT labelling evaluation on all samples."""
    print(f"🚀 Starting GT labelling evaluation")
    print(f"📁 Data directory: {data_dir}")
    print(f"🎯 Judge models: {JUDGE_MODELS}")
    print(f"🧪 Experiment name: {EXPERIMENT_NAME}")
    print(f"📊 Max samples: {num_samples or 'All'}")
    print(f"⚡ Max workers: {MAX_WORKERS}")
    
    # Load the dataset
    model_tasks = gt_labelling_rowan_modified_task_by_model(data_dir, limit=num_samples)
    
    if not model_tasks:
        print("❌ No model tasks found")
        return
    
    # Get the first model's dataset (they should all be the same)
    first_model = list(model_tasks.keys())[0]
    dataset = model_tasks[first_model].dataset
    
    print(f"📋 Found {len(dataset.samples)} samples to process")
    
    # Process each sample
    successful_samples = 0
    failed_samples = 0
    
    for i, sample in enumerate(dataset.samples):
        print(f"\n{'='*60}")
        print(f"📝 Processing sample {i+1}/{len(dataset.samples)}")
        
        # Get sample ID from metadata
        sample_id = sample.metadata.get("sample_id", f"sample_{i}")
        
        try:
            if process_sample_parallel(sample, sample_id):
                successful_samples += 1
            else:
                failed_samples += 1
        except Exception as e:
            print(f"❌ Error processing sample {sample_id}: {e}")
            failed_samples += 1
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"📊 GT LABELLING EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total samples: {len(dataset.samples)}")
    print(f"Successful samples: {successful_samples}")
    print(f"Failed samples: {failed_samples}")
    print(f"Success rate: {successful_samples/len(dataset.samples)*100:.1f}%")
    print(f"Judge models used: {len(JUDGE_MODELS)}")
    print(f"Experiment name: {EXPERIMENT_NAME}")
    
    if successful_samples == len(dataset.samples):
        print(f"🎉 All samples processed successfully!")
    else:
        print(f"⚠️ Some samples failed. Check the logs above.")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run GT labelling evaluation")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory path")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to process")
    parser.add_argument("--experiment_name", type=str, default=EXPERIMENT_NAME, help="Experiment name")
    parser.add_argument("--s3_bucket", type=str, default=S3_BUCKET, help="S3 bucket name")
    parser.add_argument("--s3_prefix", type=str, default=S3_PREFIX, help="S3 prefix")
    parser.add_argument("--max_workers", type=int, default=MAX_WORKERS, help="Max parallel workers")
    
    args = parser.parse_args()
    
    # Update global configuration
    global EXPERIMENT_NAME, S3_BUCKET, S3_PREFIX, MAX_WORKERS
    EXPERIMENT_NAME = args.experiment_name
    S3_BUCKET = args.s3_bucket
    S3_PREFIX = args.s3_prefix
    MAX_WORKERS = args.max_workers
    
    # Check required environment variables
    required_env_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        print("Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        return
    
    run_gt_labelling_evaluation(args.data_dir, args.num_samples)


if __name__ == "__main__":
    main()
