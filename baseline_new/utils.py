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
                # File doesn't exist, throw error
                raise FileNotFoundError(f"S3 file does not exist: {s3_path}")
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
        
        # Successfully wrote baseline
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
        
        # Iterate through task directories (e.g., ascii_train, mask-roleplay_train)
        for task_dir in baseline_type_dir.iterdir():
            if not task_dir.is_dir():
                continue
                
            task_name = task_dir.name
            processed_results[baseline_type][task_name] = []
            
            # Find results JSON files - look in subdirectories for the current structure
            json_files = []
            
            # First, try to find files directly in task_dir (old format)
            for json_file in task_dir.glob("results_*.json"):
                json_files.append(json_file)
            
            # Look in subdirectories for JSON files (current format: task_dir/provider/model_name/*_results.json)
            for subdir in task_dir.iterdir():
                if subdir.is_dir():
                    # Look for JSON files directly in this subdirectory
                    for json_file in subdir.glob("*_results.json"):
                        json_files.append(json_file)
                    # Also look in subdirectories of this subdirectory
                    for subsubdir in subdir.iterdir():
                        if subsubdir.is_dir():
                            for json_file in subsubdir.glob("*_results.json"):
                                json_files.append(json_file)
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        results_data = json.load(f)
                    
                    # Handle both old format (dict with metadata) and new format (list of samples)
                    if isinstance(results_data, list):
                        # New format: list of samples
                        samples = results_data
                        model_info = None
                        
                        # Try to extract model info from the first sample's metadata
                        if samples and 'metadata' in samples[0]:
                            # Look for model info in various places
                            sample_metadata = samples[0]['metadata']
                            if 'model' in sample_metadata:
                                model_info = sample_metadata['model']
                            elif 'meta' in sample_metadata and 'model' in sample_metadata['meta']:
                                model_info = sample_metadata['meta']['model']
                        
                        if not model_info:
                            # Try to infer from file path
                            path_parts = json_file.parts
                            if len(path_parts) >= 3:
                                # Extract from path like: .../openai/gpt-oss-120b/results.json
                                provider = path_parts[-3]
                                model_name = path_parts[-2]
                                model_info = f"{provider}/{model_name}"
                            else:
                                model_info = "unknown"
                        
                        # Process each sample
                        for sample in samples:
                            # Extract sample information from the new format
                            sample_metadata = sample.get('metadata', {})
                            meta = sample_metadata.get('meta', {})
                            
                            # Try both nested (meta) and direct (metadata) locations
                            sample_id = meta.get('sample_id') or sample_metadata.get('sample_id')
                            task = meta.get('task') or sample_metadata.get('task')
                            groundtruth_did_lie = meta.get('did_lie') or sample_metadata.get('original_did_lie')
                            
                            if not all([sample_id is not None, task, groundtruth_did_lie is not None]):
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
                                model_id=model_info,
                                bucket="dipika-lie-detection-data",
                                prefix="processed-data/"
                            )
                            
                            if s3_path:
                                # Check if file exists in S3
                                exists = s3_client.file_exists(s3_path)
                                
                                # Write baseline results to S3 (regardless of whether original exists)
                                baseline_written = False
                                
                                # Create baseline results structure
                                baseline_results = {
                                    'sample_id': sample_id,
                                    'task': task,
                                    'groundtruth_did_lie': groundtruth_did_lie,
                                    'baseline_type': baseline_type,
                                    'model_id': model_info,
                                    'prediction': sample.get('output', ''),
                                    'target': sample.get('target', ''),
                                    'scores': sample.get('scores', {})
                                }
                                
                                # Write to S3
                                success = write_baseline_to_s3(
                                    s3_client=s3_client,
                                    s3_path=s3_path,
                                    baseline_type=baseline_type,
                                    results=baseline_results
                                )
                                
                                if success:
                                    baseline_written = True
                                else:
                                    print(f"        ❌ Sample {sample_id}: {s3_path} (failed to write baseline)")
                                
                                sample_result = {
                                    'sample_id': sample_id,
                                    'task': task,
                                    'groundtruth_did_lie': groundtruth_did_lie,
                                    's3_path': s3_path,
                                    's3_exists': exists,
                                    'baseline_written': baseline_written
                                }
                                
                                processed_results[baseline_type][task_name].append(sample_result)
                    
                    else:
                        # Old format: dict with metadata and subtask_results
                        model_info = results_data.get('metadata', {}).get('model', '')
                        if not model_info:
                            continue
                        
                        # Parse model to get provider and model_id
                        if '/' in model_info:
                            provider, model_id = model_info.split('/', 1)
                        else:
                            provider = "unknown"
                            model_id = model_info
                        
                        # Process subtask results
                        subtask_results = results_data.get('subtask_results', {})
                        for subtask_name, subtask_data in subtask_results.items():
                            samples = subtask_data.get('samples', [])
                            
                            for sample in samples:
                                sample_id = sample.get('sample_id')
                                task = sample.get('task')
                                groundtruth_did_lie = sample.get('groundtruth_did_lie')
                                
                                if not all([sample_id, task, groundtruth_did_lie is not None]):
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
                                    prefix="processed-data/"
                                )
                                
                                if s3_path:
                                    # Check if file exists in S3
                                    exists = s3_client.file_exists(s3_path)
                                    
                                    sample_result = {
                                        'sample_id': sample_id,
                                        'task': task,
                                        'groundtruth_did_lie': groundtruth_did_lie,
                                        's3_path': s3_path,
                                        's3_exists': exists,
                                        'baseline_written': False  # Will be updated when we write baselines
                                    }
                                    
                                    processed_results[baseline_type][task_name].append(sample_result)
                                    
                                    if not exists:
                                        print(f"        ❌ Sample {sample_id}: {s3_path} (not found in S3)")
                
                except Exception as e:
                    print(f"      Error processing {json_file.name}: {e}")
                    continue
    
    # Now write baselines to S3 for all samples that exist
    print(f"\n📝 Writing baselines to S3 for existing samples...")
    total_written = 0
    total_failed = 0
    
    for baseline_type, tasks in processed_results.items():
        for task_name, samples in tasks.items():
            for sample in samples:
                if sample['s3_exists']:
                    # For the new format, we need to reconstruct the sample data
                    # The baseline writing function expects the original sample structure
                    baseline_written = write_baseline_to_s3(
                        s3_client=s3_client,
                        s3_path=sample['s3_path'],
                        baseline_type=baseline_type,
                        results={
                            'sample_id': sample['sample_id'],
                            'task': sample['task'],
                            'groundtruth_did_lie': sample['groundtruth_did_lie']
                        }
                    )
                    sample['baseline_written'] = baseline_written
                    if baseline_written:
                        total_written += 1
                    else:
                        total_failed += 1
                        print(f"    ❌ Failed to write baseline to S3 URI: {sample['s3_path']}")
    
    print(f"✅ Successfully wrote {total_written} baselines to S3")
    if total_failed > 0:
        print(f"❌ Failed to write {total_failed} baselines to S3")
    
    return processed_results


def map_sample_to_s3_path(sample, model_id, bucket="dipika-lie-detection-data", prefix="processed-data/"):
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
    task_type, domain = s3_client._parse_task_name(task)
    
    # Clean names using S3SampleClient's actual method
    clean_domain = s3_client._clean_name(domain)
    clean_task_type = s3_client._clean_name(task_type)
    
    # Create path as task_type/domain (matches the expected S3 structure)
    # For ascii_train -> ascii/train, mask_continuations -> mask/continuations
    task_path = f"{clean_task_type}/{clean_domain}"
    
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
    
    # Clean model_id (replace hyphens with underscores)
    clean_model_id = model_id.replace("-", "_")
    
    # Construct S3 path using cleaned model_id
    s3_path = f"s3://{bucket}/{prefix}{clean_model_id}/{task_path}/{filename}"
    
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
