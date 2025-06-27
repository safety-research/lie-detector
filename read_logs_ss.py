import json
import glob
import os

# Get all JSON log files for self-sycophancy
log_dir = "/mnt/nvme3/dipika/lie-detector/ss_run2"
log_files = glob.glob(log_dir + "/*.json")

data = []
error_stats = {
    'total_samples_processed': 0,
    'samples_with_no_choices': 0,
    'samples_with_processing_errors': 0,
    'files_with_errors': 0,
    'error_sample_ids': []
}

for log_file in log_files:
    try:
        with open(log_file, 'r') as f:
            log_data = json.load(f)
        
        model_name = log_data['eval']['model']
        task_name = log_data['eval']['task']
        task_id = log_data['eval']['task_id']
        
        print(f"Processing {task_name} with {model_name}")
        
        for sample in log_data['samples']:
            error_stats['total_samples_processed'] += 1
            
            try:
                # Extract basic sample info
                sample_id = sample['id']
                question = sample['input']
                target = sample['target']
                
                # Extract the model's answer from the output
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
                
                # Extract scores and flatten them to match mask schema
                scores = {}
                score_explanations = {}
                for scorer_name, scorer_result in sample['scores'].items():
                    scores[scorer_name] = scorer_result['value']
                    score_explanations[scorer_name] = scorer_result.get('explanation', '')
                
                # Extract metadata
                metadata = sample.get('metadata', {})
                chosen_option = metadata.get('chosen_option')
                all_options = metadata.get('all_options', [])
                rejected = metadata.get('rejected', '')
                
                # Create a unified schema similar to mask tasks
                data.append({
                    'task': task_name,
                    'task_id': task_id,
                    'model': model_name,
                    'sample_id': sample_id,
                    'question': question,
                    'answer': answer_text,
                    'target': target,
                    # Flatten scores to match mask schema
                    'score': scores.get('answer_accuracy_scorer', 'N/A'),  # Primary accuracy score
                    'ground_truth': f"Correct option: {chosen_option if chosen_option else 'N/A'}",  # Similar to mask ground_truth
                    # Additional behavioral scores
                    'sycophancy_score': scores.get('self_sycophancy_scorer', 'N/A'),
                    'skepticism_score': scores.get('self_skepticism_scorer', 'N/A'),
                    # Metadata for context
                    'chosen_option': chosen_option,
                    'all_options': all_options,
                    'rejected': rejected,
                    # Score explanations
                    'score_explanations': score_explanations
                })
                
            except Exception as e:
                error_stats['samples_with_processing_errors'] += 1
                error_stats['error_sample_ids'].append(sample.get('id', 'unknown'))
                print(f"Error processing sample {sample.get('id', 'unknown')}: {e}")
                continue
                
    except Exception as e:
        error_stats['files_with_errors'] += 1
        print(f"Error processing file {log_file}: {e}")
        continue

# Save the parsed data
output_file = "parsed_self_sycophancy_jsons.json"
with open(output_file, 'w') as f:
    json.dump(data, f, indent=2)

# Save error stats
error_file = "error_stats_ss.json"
with open(error_file, 'w') as f:
    json.dump(error_stats, f, indent=2)

# Print summary
print(f"\n=== SELF-SYCOPHANCY PROCESSING SUMMARY ===")
print(f"Successfully parsed: {len(data)} samples")
print(f"Total samples processed: {error_stats['total_samples_processed']}")
print(f"Samples with no choices: {error_stats['samples_with_no_choices']}")
print(f"Samples with processing errors: {error_stats['samples_with_processing_errors']}")
print(f"Files with errors: {error_stats['files_with_errors']}")
print(f"Success rate: {(len(data) / error_stats['total_samples_processed'] * 100):.1f}%")
print(f"Error sample IDs: {error_stats['error_sample_ids']}")

print(f"\nParsed {len(data)} samples from {len(log_files)} files to {output_file}")
print(f"Error stats saved to {error_file}")

# Print sample data structure
if data:
    print(f"\n=== UNIFIED SCHEMA SAMPLE ===")
    sample = data[0]
    print(f"Task: {sample['task']}")
    print(f"Task ID: {sample['task_id']}")
    print(f"Model: {sample['model']}")
    print(f"Sample ID: {sample['sample_id']}")
    print(f"Question: {sample['question']}")
    print(f"Target: {sample['target']}")
    print(f"Answer: {sample['answer'][:100]}...")
    print(f"Primary Score: {sample['score']}")
    print(f"Ground Truth: {sample['ground_truth']}")
    print(f"Sycophancy Score: {sample['sycophancy_score']}")
    print(f"Skepticism Score: {sample['skepticism_score']}")
    print(f"Chosen Option: {sample['chosen_option']}")

# Print task breakdown
task_counts = {}
for item in data:
    task = item['task']
    task_counts[task] = task_counts.get(task, 0) + 1

print(f"\n=== TASK BREAKDOWN ===")
for task, count in task_counts.items():
    print(f"{task}: {count} samples")

# Print score breakdown
print(f"\n=== SCORE BREAKDOWN ===")
score_breakdown = {}
for item in data:
    score = item['score']
    if score not in score_breakdown:
        score_breakdown[score] = 0
    score_breakdown[score] += 1
print(f"Primary Score: {score_breakdown}")

sycophancy_breakdown = {}
for item in data:
    score = item['sycophancy_score']
    if score not in sycophancy_breakdown:
        sycophancy_breakdown[score] = 0
    sycophancy_breakdown[score] += 1
print(f"Sycophancy Score: {sycophancy_breakdown}")

skepticism_breakdown = {}
for item in data:
    score = item['skepticism_score']
    if score not in skepticism_breakdown:
        skepticism_breakdown[score] = 0
    skepticism_breakdown[score] += 1
print(f"Skepticism Score: {skepticism_breakdown}") 