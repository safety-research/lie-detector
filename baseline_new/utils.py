import os
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path to import from common
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import S3SampleClient to use its exact logic
from common.s3_sample_client import S3SampleClient


def write_baseline_to_s3(s3_client: S3SampleClient, s3_path: str, baseline_type: str, results: Dict[str, Any]) -> bool:
    """
    Write baseline results to an existing S3 JSON file.
    
    Args:
        s3_client: S3SampleClient instance
        s3_path: S3 path (e.g., "s3://bucket/key")
        baseline_type: Type of baseline (e.g., "llama_chat")
        results: Results data to store
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Parse S3 path
        s3_uri = s3_path.replace('s3://', '')
        bucket_name, key = s3_uri.split('/', 1)
        
        # Read existing JSON from S3
        try:
            response = s3_client.s3_client.get_object(Bucket=bucket_name, Key=key)
            existing_data = json.loads(response['Body'].read().decode('utf-8'))
        except Exception as e:
            if 'NoSuchKey' in str(e):
                # File doesn't exist, create new structure
                existing_data = {}
            else:
                print(f"      Error reading S3 file: {e}")
                return False
        
        # Ensure baseline key exists
        if 'baseline' not in existing_data:
            existing_data['baseline'] = {}
        
        # Add/update baseline results
        existing_data['baseline'][baseline_type] = results
        
        # Write updated JSON back to S3
        updated_json = json.dumps(existing_data, indent=2)
        s3_client.s3_client.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=updated_json.encode('utf-8'),
            ContentType='application/json'
        )
        
        print(f"        ✅ Successfully wrote baseline '{baseline_type}' to S3")
        return True
        
    except Exception as e:
        print(f"      Error writing baseline to S3: {e}")
        return False


def post_process_results(results_dir: str = "/Users/dipikakhullar/Desktop/lie-detector/baseline_new/results") -> Dict[str, Any]:
    """
    Post-process results directory to map samples to their S3 paths and check existence.
    
    Args:
        results_dir (str): Path to the results directory
        
    Returns:
        Dict containing S3 URI mappings and sample results
    """
    s3_client = S3SampleClient()
    processed_results = {}
    
    # Iterate through baseline_type directories
    for baseline_type_dir in Path(results_dir).iterdir():
        if not baseline_type_dir.is_dir():
            continue
            
        baseline_type = baseline_type_dir.name
        processed_results[baseline_type] = {}
        
        print(f"Processing baseline type: {baseline_type}")
        
        # Iterate through task directories (e.g., ascii_train, mask-roleplay_train)
        for task_dir in baseline_type_dir.iterdir():
            if not task_dir.is_dir():
                continue
                
            task_name = task_dir.name
            processed_results[baseline_type][task_name] = []
            
            print(f"  Processing task: {task_name}")
            
            # Find results JSON files
            for json_file in task_dir.glob("results_*.json"):
                print(f"    Processing file: {json_file.name}")
                
                try:
                    with open(json_file, 'r') as f:
                        results_data = json.load(f)
                    
                    # Extract model information
                    model_info = results_data.get('metadata', {}).get('model', '')
                    if not model_info:
                        print(f"      Warning: No model info found in {json_file.name}")
                        continue
                    
                    # Parse model to get provider and model_id
                    # Format: "openrouter/google/gemma-3-4b-it" -> provider="openrouter", model_id="google/gemma-3-4b-it"
                    if '/' in model_info:
                        provider, model_id = model_info.split('/', 1)
                    else:
                        provider = "unknown"
                        model_id = model_info
                    
                    print(f"      Model: {model_id} (Provider: {provider})")
                    
                    # Process subtask results
                    subtask_results = results_data.get('subtask_results', {})
                    for subtask_name, subtask_data in subtask_results.items():
                        samples = subtask_data.get('samples', [])
                        
                        for sample in samples:
                            sample_id = sample.get('sample_id')
                            task = sample.get('task')
                            groundtruth_did_lie = sample.get('groundtruth_did_lie')
                            
                            if not all([sample_id, task, groundtruth_did_lie is not None]):
                                print(f"        Warning: Missing required fields for sample {sample_id}")
                                continue
                            
                            # Create sample dict for map_sample_to_s3_path
                            sample_dict = {
                                'meta': {
                                    'task': task,
                                    'sample_id': sample_id,
                                    'did_lie': groundtruth_did_lie
                                }
                            }
                            
                            # Map to S3 path
                            s3_path = map_sample_to_s3_path(
                                sample=sample_dict,
                                model_id=model_id,
                                bucket="dipika-lie-detection-data",
                                prefix="processed-data-test/"
                            )
                            
                            if not s3_path:
                                print(f"        Warning: Could not map sample {sample_id} to S3 path")
                                continue
                            
                            # Check if S3 path exists
                            s3_uri = s3_path.replace('s3://', '')
                            bucket_name, key = s3_uri.split('/', 1)
                            
                            try:
                                s3_exists = s3_client.s3_client.head_object(Bucket=bucket_name, Key=key)
                                exists = True
                            except Exception as e:
                                if 'NoSuchKey' in str(e):
                                    exists = False
                                else:
                                    print(f"        Error checking S3 existence: {e}")
                                    exists = False
                            
                            # Store processed result (without baseline writing for now)
                            processed_sample = {
                                's3_uri': s3_path,
                                's3_exists': exists,
                                'sample_id': sample_id,
                                'task': task,
                                'groundtruth_did_lie': groundtruth_did_lie,
                                'model_id': model_id,
                                'provider': provider,
                                'baseline_type': baseline_type,
                                'results': sample
                            }
                            
                            processed_results[baseline_type][task_name].append(processed_sample)
                            
                            print(f"        Sample {sample_id}: S3 path exists: {exists}")
                            
                except Exception as e:
                    print(f"      Error processing {json_file.name}: {e}")
                    continue
    
    # Now write baselines to S3 for all samples that exist
    print(f"\n📝 Writing baselines to S3 for existing samples...")
    for baseline_type, tasks in processed_results.items():
        for task_name, samples in tasks.items():
            for sample in samples:
                if sample['s3_exists']:
                    print(f"  Writing baseline '{baseline_type}' to {sample['sample_id']}")
                    baseline_written = write_baseline_to_s3(
                        s3_client=s3_client,
                        s3_path=sample['s3_uri'],
                        baseline_type=baseline_type,
                        results=sample['results']
                    )
                    sample['baseline_written'] = baseline_written
                    if baseline_written:
                        print(f"    ✅ Successfully wrote baseline to S3 URI: {sample['s3_uri']}")
                    else:
                        print(f"    ❌ Failed to write baseline to S3 URI: {sample['s3_uri']}")
    
    return processed_results


def map_sample_to_s3_path(sample, model_id, bucket="dipika-lie-detection-data", prefix="processed-data-test/"):
    """
    Maps a sample to its corresponding S3 path based on the data structure. the format is [domain-separated-by-hyphens]_[task_separated_by_underscores]
    
    Args:
        sample (dict or object): The sample data containing meta information
        provider (str): The provider (e.g., 'google', 'openai', 'meta_llama')
        model (str): The model name (e.g., 'gemma_3_4b_it', 'gpt_4o_mini')
        fold_name (str): The fold name ('train' or 'val')
        bucket (str): S3 bucket name
        prefix (str): S3 prefix for the data
    
    Returns:
        str: The S3 path for the sample
    """
    # Extract task information from sample - handle both dict and object formats
    if isinstance(sample, dict):
        # Dictionary format: check meta first, then direct keys
        if 'meta' in sample and isinstance(sample['meta'], dict):
            task = sample['meta'].get('task')
            sample_id = sample['meta'].get('sample_id')
        else:
            # Direct keys
            task = sample.get('task')
            sample_id = sample.get('sample_id')
    elif hasattr(sample, 'task') and hasattr(sample, 'sample_id'):
        # Object format: sample.task and sample.sample_id directly
        task = getattr(sample, 'task', None)
        sample_id = getattr(sample, 'sample_id', None)
    elif hasattr(sample, 'meta'):
        # Object format: sample.meta.task
        task = getattr(sample.meta, 'task', None)
        sample_id = getattr(sample.meta, 'sample_id', None)
    else:
        # Fallback: try to get as attributes
        task = getattr(sample, 'task', None)
        sample_id = getattr(sample, 'id', None)
    
    if not task or not sample_id:
        print(f"[map_sample_to_s3_path] Warning: Could not extract task or sample_id from sample")
        return None
    
    # Use actual S3SampleClient methods to ensure exact matching
    s3_client = S3SampleClient()
    
    # Parse task using S3SampleClient's actual method
    domain, task_type = s3_client._parse_task_name(task)
    
    # Clean names using S3SampleClient's actual method
    clean_domain = s3_client._clean_name(domain)
    clean_task_type = s3_client._clean_name(task_type)
    
    # Create path as domain/task_type (matches the expected S3 structure)
    # For ascii_train -> ascii/train, mask_continuations -> mask/continuations
    task_path = f"{clean_domain}/{clean_task_type}"
    
    # Generate filename based on sample_id to match S3SampleClient logic
    # The S3SampleClient uses truth tags (t_ or f_) and generate_sample_id() method
    
    # Extract did_lie from sample to determine truth tag
    did_lie = sample['meta']['did_lie']
    
    # Determine truth tag (matches S3SampleClient line 118)
    truth_tag = "t_" if did_lie else "f_"
    
    # Generate clean sample ID using S3SampleClient's actual method
    clean_sample_id = s3_client.generate_sample_id(sample_id)
    
    # Create filename: {truth_tag}{clean_sample_id}.json (matches S3SampleClient line 121)
    filename = f"{truth_tag}{clean_sample_id}.json"
    
    # Construct S3 path using model_id directly
    s3_path = f"s3://{bucket}/{prefix}{model_id}/{task_path}/{filename}"
    
    return s3_path

def extract_provider_model_from_path(local_path):
    """
    Extract provider and model from a local file path.
    
    Args:
        local_path (str): Local path like ".data/google/gemma_3_4b_it/domain/train.jsonl"
        
    Returns:
        tuple: (provider, model) or (None, None) if cannot be extracted
        
    Example:
        >>> extract_provider_model_from_path(".data/google/gemma_3_4b_it/self-sycophancy/train.jsonl")
        ('google', 'gemma_3_4b_it')
    """
    from pathlib import Path
    
    try:
        path_parts = Path(local_path).parts
        
        # Find .data directory index
        data_index = None
        for i, part in enumerate(path_parts):
            if part == '.data' or part == 'data':
                data_index = i
                break
        
        if data_index is None or len(path_parts) < data_index + 3:
            return None, None
        
        provider = path_parts[data_index + 1]
        model = path_parts[data_index + 2]
        
        return provider, model
        
    except (IndexError, AttributeError):
        return None, None
