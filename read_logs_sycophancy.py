import json
import glob
import os
import pandas as pd
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import re

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

# Get all JSON log files for sycophancy tasks
log_dir = "/mnt/nvme3/dipika/lie-detector/sycophancy_logs_full_run5"  # Change this to your log directory
log_files = get_latest_file_versions(log_dir)

data = []
error_stats = {
    'total_files_processed': 0,
    'files_with_errors': 0,
    'total_samples_processed': 0,
    'successful_samples': 0,
    'failed_samples': 0,
    'samples_with_processing_errors': 0,
    'error_sample_ids': []
}

for log_file in log_files:
    try:
        with open(str(log_file), 'r') as f:
            log_data = json.load(f)
        
        error_stats['total_files_processed'] += 1
        
        model_name = log_data['eval']['model']
        task_name = log_data['eval']['task']
        task_id = log_data['eval']['task_id']
        
        print(f"Processing {task_name} with {model_name}")
        
        # Check if the log has samples (successful runs) or just errors
        if 'samples' not in log_data:
            print(f"  No samples found in {log_file}")
            error_stats['files_with_errors'] += 1
            continue
            
        for sample in log_data['samples']:
            error_stats['total_samples_processed'] += 1
            
            try:
                # Check if sample has an error
                if 'error' in sample:
                    error_stats['failed_samples'] += 1
                    error_stats['error_sample_ids'].append(sample.get('id', 'unknown'))
                    print(f"  Sample {sample.get('id', 'unknown')} failed: {sample['error'].get('message', 'Unknown error')}")
                    continue
                
                # Extract basic sample info
                sample_id = sample['id']
                question = sample['input']
                target = sample['target']
                
                # Extract the model's answer from the output
                if 'output' not in sample or 'choices' not in sample['output'] or len(sample['output']['choices']) == 0:
                    error_stats['samples_with_processing_errors'] += 1
                    error_stats['error_sample_ids'].append(sample_id)
                    print(f"  Sample {sample_id} has no output choices")
                    continue
                
                answer_content = sample['output']['choices'][0]['message']['content']
                
                # Handle both string and list content formats
                if isinstance(answer_content, str):
                    answer_text = answer_content
                elif isinstance(answer_content, list):
                    # The answer is a list of content blocks, we want the text part
                    answer_text = ""
                    for content_block in answer_content:
                        if content_block['type'] == 'text':
                            answer_text = content_block['text']
                            break
                else:
                    answer_text = str(answer_content)
                
                # Extract scores from sycophancy scorer
                scores = {}
                score_explanations = {}
                if 'scores' in sample:
                    for scorer_name, scorer_result in sample['scores'].items():
                        if isinstance(scorer_result, dict) and 'value' in scorer_result:
                            scores[scorer_name] = scorer_result['value']
                            score_explanations[scorer_name] = scorer_result.get('metadata', {})
                
                # Extract metadata
                metadata = sample.get('metadata', {})
                correct_answer = metadata.get('correct_answer', 'N/A')
                incorrect_answer = metadata.get('incorrect_answer', 'N/A')
                question_text = metadata.get('question_text', question)
                prompt_type = metadata.get('prompt_type', 'N/A')
                
                # Get control response if available
                control_response = metadata.get('control_response', 'N/A')
                influenced_response = metadata.get('influenced_response', answer_text)
                
                # Create a unified schema similar to other tasks
                data.append({
                    'task': task_name,
                    'task_id': task_id,
                    'model': model_name,
                    'sample_id': sample_id,
                    'question': question_text,
                    'answer': answer_text,
                    'target': target,
                    # Primary sycophancy score
                    'sycophancy_score': scores.get('sycophancy_control_comparison_scorer', 'N/A'),
                    # Ground truth info
                    'correct_answer': correct_answer,
                    'incorrect_answer': incorrect_answer,
                    'prompt_type': prompt_type,
                    # Response comparison
                    'influenced_response': influenced_response,
                    'control_response': control_response,
                    # Score explanations
                    'score_explanations': score_explanations,
                    # Additional metadata
                    'metadata': metadata
                })
                
                error_stats['successful_samples'] += 1
                
            except Exception as e:
                error_stats['samples_with_processing_errors'] += 1
                error_stats['error_sample_ids'].append(sample.get('id', 'unknown'))
                print(f"Error processing sample {sample.get('id', 'unknown')}: {e}")
                continue
                
    except Exception as e:
        error_stats['files_with_errors'] += 1
        print(f"Error processing file {log_file}: {e}")
        continue

# Print summary statistics
print("\n" + "="*50)
print("PROCESSING SUMMARY")
print("="*50)
print(f"Total files processed: {error_stats['total_files_processed']}")
print(f"Files with errors: {error_stats['files_with_errors']}")
print(f"Total samples processed: {error_stats['total_samples_processed']}")
print(f"Successful samples: {error_stats['successful_samples']}")
print(f"Failed samples: {error_stats['failed_samples']}")
print(f"Samples with processing errors: {error_stats['samples_with_processing_errors']}")
print(f"Success rate: {error_stats['successful_samples']/max(1, error_stats['total_samples_processed'])*100:.1f}%")

# Save the data
if data:
    # Save as JSON with consistent naming
    json_filename = "parsed_sycophancy_jsons.json"
    with open(json_filename, 'w') as f:
        json.dump({
            'data': data,
            'error_stats': error_stats,
            'processing_timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }, f, indent=2)
    print(f"\nData saved to: {json_filename}")
    
    # Print some sample data
    print(f"\nSample data (first 3 entries):")
    for i, item in enumerate(data[:3]):
        print(f"\nEntry {i+1}:")
        print(f"  Task: {item['task']}")
        print(f"  Model: {item['model']}")
        print(f"  Sample ID: {item['sample_id']}")
        print(f"  Sycophancy Score: {item['sycophancy_score']}")
        print(f"  Question: {item['question'][:100]}...")
        print(f"  Answer: {item['answer'][:100]}...")
    
    # Print sycophancy score distribution
    scores = [item['sycophancy_score'] for item in data if item['sycophancy_score'] != 'N/A']
    if scores:
        print(f"\nSycophancy Score Distribution:")
        score_counts = {}
        for score in scores:
            score_counts[score] = score_counts.get(score, 0) + 1
        for score, count in sorted(score_counts.items()):
            print(f"  {score}: {count} samples ({count/len(scores)*100:.1f}%)")
    
else:
    print("\nNo successful data found to save!") 