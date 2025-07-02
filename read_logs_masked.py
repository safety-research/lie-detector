import json
import glob
import os
import re
from pathlib import Path
from collections import defaultdict

def get_latest_file_versions(json_dir: str):
    """
    Given a directory with JSON logs, return a list of full paths to the most recent
    version of each unique file ID (based on timestamp prefix in filename).
    """
    path = Path(json_dir)
    assert path.exists() and path.is_dir(), f"Invalid directory: {json_dir}"

    pattern = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{2}-\d{2})_(.*)\.json$")
    file_groups = defaultdict(list)

    for f in path.glob("*.json"):
        match = pattern.search(f.name)
        if not match:
            continue
        timestamp, file_id = match.groups()
        file_groups[file_id].append((timestamp, f.resolve()))

    latest_files = []
    for file_id, files in file_groups.items():
        # Sort by timestamp descending and take the latest
        files.sort(reverse=True)
        latest_files.append(files[0][1])  # keep the Path object (resolved full path)

    return latest_files

# Get all JSON log files (latest versions only)
log_dir = "/mnt/nvme3/dipika/lie-detector/masked_logs_full_run4"
print(f"Getting latest file versions from {log_dir}...")
log_files = get_latest_file_versions(log_dir)
print(f"Found {len(log_files)} latest file versions")

data = []
error_stats = {
    'total_samples_processed': 0,
    'samples_with_no_choices': 0,
    'samples_with_no_scorer': 0,
    'samples_with_processing_errors': 0,
    'files_with_errors': 0,
    'error_sample_ids': []
}

for log_file_path in log_files:
    try:
        with open(log_file_path, 'r') as f:
            log_data = json.load(f)
        
        model_name = log_data['eval']['model']
        task_name = log_data['eval']['task']
        task_id = log_data['eval']['task_id']
        
        for sample in log_data['samples']:
            error_stats['total_samples_processed'] += 1
            
            try:
                # Check if choices array exists and has content
                if 'output' not in sample or 'choices' not in sample['output'] or len(sample['output']['choices']) == 0:
                    sample_id = sample.get('id', 'unknown')
                    print(f"Warning: No choices found in sample {sample_id} in {log_file_path.name}")
                    error_stats['samples_with_no_choices'] += 1
                    error_stats['error_sample_ids'].append(sample_id)
                    continue
                
                # Check if scorer exists
                if 'scores' not in sample or 'mask_combined_scorer' not in sample['scores']:
                    sample_id = sample.get('id', 'unknown')
                    print(f"Warning: No mask_combined_scorer found in sample {sample_id} in {log_file_path.name}")
                    error_stats['samples_with_no_scorer'] += 1
                    error_stats['error_sample_ids'].append(sample_id)
                    continue
                    
                question = sample['input']
                answer = sample['output']['choices'][0]['message']['content']
                target = sample['target']
                score = sample['scores']['mask_combined_scorer']['value']
                formatted_ground_truth = sample['metadata']['formatted_ground_truth']
                sample_id = sample['scores']['mask_combined_scorer'].get('sample_id', sample.get('id', 'unknown'))
                
                # Extract the full trace for this sample - create conversation history
                # Start with the input messages
                conversation_trace = sample.get('messages', [])
                
                # Add the assistant's response to complete the conversation
                if 'output' in sample and 'choices' in sample['output'] and len(sample['output']['choices']) > 0:
                    assistant_message = sample['output']['choices'][0]['message']
                    conversation_trace.append(assistant_message)
                
                # The trace is now a list of message objects (user + assistant pairs)
                full_trace = conversation_trace
                
                data.append({
                    'task': task_name,
                    'task_id': task_id,
                    'model': model_name,
                    'sample_id': sample_id,
                    'question': question,
                    'answer': answer,
                    'target': target,
                    'score': score,
                    "ground_truth": formatted_ground_truth,
                    'trace': full_trace  # Full trace for prefilling to another model
                })
                
            except Exception as e:
                sample_id = sample.get('id', 'unknown')
                print(f"Error processing sample {sample_id} in {log_file_path.name}: {e}")
                error_stats['samples_with_processing_errors'] += 1
                error_stats['error_sample_ids'].append(sample_id)
                continue
                
    except Exception as e:
        print(f"Error processing file {log_file_path.name}: {e}")
        error_stats['files_with_errors'] += 1
        continue

# Write data to file
with open('masked_parsed_logs.json', 'w') as f:
    json.dump(data, f, indent=2)

# Print summary
print(f"\n=== PROCESSING SUMMARY (Latest Versions Only) ===")
print(f"Successfully parsed: {len(data)} samples")
print(f"Total samples processed: {error_stats['total_samples_processed']}")
print(f"Samples with no choices: {error_stats['samples_with_no_choices']}")
print(f"Samples with no scorer: {error_stats['samples_with_no_scorer']}")
print(f"Samples with processing errors: {error_stats['samples_with_processing_errors']}")
print(f"Files with errors: {error_stats['files_with_errors']}")
print(f"Success rate: {len(data) / error_stats['total_samples_processed'] * 100:.1f}%")
print(f"Error sample IDs: {error_stats['error_sample_ids']}")
print(f"\nParsed {len(data)} samples from {len(log_files)} latest files to masked_parsed_logs.json")
print(f"Error stats saved to error_stats.json")
print(f"\nEach sample now includes a 'trace' field with full trace data for prefilling to other models") 