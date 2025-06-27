import json
import glob
import os

# Get all JSON log files
log_dir = "/mnt/nvme3/dipika/lie-detector/masked_logs_run1"
log_files = glob.glob(log_dir +"/*json")
# log_files = glob.glob("logs/*.json")

data = []
error_stats = {
    'total_samples_processed': 0,
    'samples_with_no_choices': 0,
    'samples_with_processing_errors': 0,
    'files_with_errors': 0,
    'error_sample_ids': []
}

for log_file in log_files:
    with open(log_file, 'r') as f:
        log_data = json.load(f)
    
    model_name = log_data['eval']['model']
    task_name = log_data['eval']['task']
    task_id = log_data['eval']['task_id']
    
    for sample in log_data['samples']:
        error_stats['total_samples_processed'] += 1
        
        # Check if choices array exists and has content
        if 'output' not in sample or 'choices' not in sample['output'] or len(sample['output']['choices']) == 0:
            sample_id = sample.get('id', 'unknown')
            print(f"Warning: No choices found in sample {sample_id} in {log_file}")
            error_stats['samples_with_no_choices'] += 1
            error_stats['error_sample_ids'].append(sample_id)
            continue
            
        question = sample['input']
        answer = sample['output']['choices'][0]['message']['content']
        target = sample['target']
        score = sample['scores']['mask_combined_scorer']['value']
        formatted_ground_truth = sample['metadata']['formatted_ground_truth']
        sample_id = sample['scores']['mask_combined_scorer'].get('sample_id', sample.get('id', 'unknown'))
        
        # # Extract logprobs if available
        # logprobs = []
        # if 'logprobs' in sample['output']['choices'][0]:
        #     for token_data in sample['output']['choices'][0]['logprobs']['content']:
        #         logprobs.append({
        #             'token': token_data['token'],
        #             'logprob': token_data['logprob']
        #         })
        
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
            # 'logprobs': logprobs
        })

# Write data to file
with open('parsed_jsons.json', 'w') as f:
    json.dump(data, f, indent=2)

# Write error stats to file
with open('error_stats.json', 'w') as f:
    json.dump(error_stats, f, indent=2)

# Print summary
print(f"\n=== PROCESSING SUMMARY ===")
print(f"Successfully parsed: {len(data)} samples")
print(f"Total samples processed: {error_stats['total_samples_processed']}")
print(f"Samples with no choices: {error_stats['samples_with_no_choices']}")
print(f"Samples with processing errors: {error_stats['samples_with_processing_errors']}")
print(f"Files with errors: {error_stats['files_with_errors']}")
print(f"Success rate: {len(data) / error_stats['total_samples_processed'] * 100:.1f}%")
print(f"Error sample IDs: {error_stats['error_sample_ids']}")
print(f"\nParsed {len(data)} samples from {len(log_files)} files to parsed_jsons.json")
print(f"Error stats saved to error_stats.json")