import json
import glob
import os

# Get all JSON log files
log_files = glob.glob("logs/*.json")

data = []

for log_file in log_files:
    with open(log_file, 'r') as f:
        log_data = json.load(f)
    
    model_name = log_data['eval']['model']
    task_name = log_data['eval']['task']
    task_id = log_data['eval']['task_id']
    
    for sample in log_data['samples']:
        question = sample['input']
        answer = sample['output']['choices'][0]['message']['content']
        target = sample['target']
        score = sample['scores']['mask_combined_scorer']['value']
        formatted_ground_truth = sample['metadata']['formatted_ground_truth']
        
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
            'question': question,
            'answer': answer,
            'target': target,
            'score': score,
            "ground_truth": formatted_ground_truth, 
            # 'logprobs': logprobs
        })

# Write to file
with open('parsed_jsons.json', 'w') as f:
    json.dump(data, f, indent=2)

print(f"Parsed {len(data)} samples from {len(log_files)} files to parsed_jsons.json")