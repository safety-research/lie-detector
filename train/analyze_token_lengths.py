import json
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from collections import defaultdict

def analyze_token_lengths(data_path, model_name='google/gemma-3-4b-it'):
    """Analyze token lengths in training data to determine optimal max_length"""
    
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading data from: {data_path}")
    
    # Load all examples
    examples = []
    with open(data_path, 'r') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    
    print(f"Loaded {len(examples)} examples")
    
    # Analyze token lengths
    conversation_lengths = []
    completion_lengths = []
    total_lengths = []
    
    for i, example in enumerate(examples):
        if i % 1000 == 0:
            print(f"Processing example {i}/{len(examples)}")
        
        messages = example["messages"]
        completion = example["completion"]
        
        # Tokenize conversation (without completion)
        conversation_tokens = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )[0].tolist()
        
        # Tokenize completion
        completion_tokens = tokenizer(
            completion,
            truncation=False,
            padding=False,
            add_special_tokens=False
        )["input_ids"]
        
        conversation_lengths.append(len(conversation_tokens))
        completion_lengths.append(len(completion_tokens))
        total_lengths.append(len(conversation_tokens) + len(completion_tokens))
    
    # Calculate statistics
    print("\n" + "="*80)
    print("TOKEN LENGTH ANALYSIS")
    print("="*80)
    
    print(f"\nConversation lengths (without completion):")
    print(f"  Min: {np.min(conversation_lengths)}")
    print(f"  Max: {np.max(conversation_lengths)}")
    print(f"  Mean: {np.mean(conversation_lengths):.1f}")
    print(f"  Median: {np.median(conversation_lengths):.1f}")
    print(f"  Std: {np.std(conversation_lengths):.1f}")
    print(f"  P50: {np.percentile(conversation_lengths, 50):.1f}")
    print(f"  P75: {np.percentile(conversation_lengths, 75):.1f}")
    print(f"  P90: {np.percentile(conversation_lengths, 90):.1f}")
    print(f"  P95: {np.percentile(conversation_lengths, 95):.1f}")
    print(f"  P99: {np.percentile(conversation_lengths, 99):.1f}")
    
    print(f"\nCompletion lengths:")
    print(f"  Min: {np.min(completion_lengths)}")
    print(f"  Max: {np.max(completion_lengths)}")
    print(f"  Mean: {np.mean(completion_lengths):.1f}")
    print(f"  Most common: {max(set(completion_lengths), key=completion_lengths.count)}")
    
    print(f"\nTotal lengths (conversation + completion):")
    print(f"  Min: {np.min(total_lengths)}")
    print(f"  Max: {np.max(total_lengths)}")
    print(f"  Mean: {np.mean(total_lengths):.1f}")
    print(f"  Median: {np.median(total_lengths):.1f}")
    print(f"  P50: {np.percentile(total_lengths, 50):.1f}")
    print(f"  P75: {np.percentile(total_lengths, 75):.1f}")
    print(f"  P90: {np.percentile(total_lengths, 90):.1f}")
    print(f"  P95: {np.percentile(total_lengths, 95):.1f}")
    print(f"  P99: {np.percentile(total_lengths, 99):.1f}")
    
    # Recommendations
    print(f"\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    p95_total = np.percentile(total_lengths, 95)
    p99_total = np.percentile(total_lengths, 99)
    p95_conv = np.percentile(conversation_lengths, 95)
    p99_conv = np.percentile(conversation_lengths, 99)
    
    print(f"\nFor max_length in training config:")
    print(f"  Conservative (P95): {int(p95_total)} tokens")
    print(f"  Aggressive (P99): {int(p99_total)} tokens")
    print(f"  Very aggressive (P99 + buffer): {int(p99_total + 100)} tokens")
    
    print(f"\nFor conversation-only max_length:")
    print(f"  Conservative (P95): {int(p95_conv)} tokens")
    print(f"  Aggressive (P99): {int(p99_conv)} tokens")
    
    # Memory considerations
    print(f"\nMemory considerations:")
    print(f"  With batch_size=8 and max_length={int(p95_total)}:")
    print(f"    Approximate GPU memory: {8 * int(p95_total) * 2 * 4 / 1024:.1f} MB (rough estimate)")
    print(f"  With batch_size=4 and max_length={int(p99_total)}:")
    print(f"    Approximate GPU memory: {4 * int(p99_total) * 2 * 4 / 1024:.1f} MB (rough estimate)")
    
    # Distribution analysis
    print(f"\n" + "="*80)
    print("DISTRIBUTION ANALYSIS")
    print("="*80)
    
    # Count examples that would be truncated at different lengths
    lengths_to_test = [512, 1024, 1536, 2048, 2560, 3072, 4096]
    
    print(f"\nExamples that would be truncated at different max_lengths:")
    for max_len in lengths_to_test:
        truncated_count = sum(1 for length in total_lengths if length > max_len)
        percentage = (truncated_count / len(total_lengths)) * 100
        print(f"  max_length={max_len}: {truncated_count} examples ({percentage:.1f}%)")
    
    return {
        'conversation_lengths': conversation_lengths,
        'completion_lengths': completion_lengths,
        'total_lengths': total_lengths,
        'p95_total': p95_total,
        'p99_total': p99_total,
        'p95_conv': p95_conv,
        'p99_conv': p99_conv
    }

if __name__ == "__main__":
    data_path = "/workspace/lie-detector/organized_balanced_training_20250722_135859/general_train_test_split_llama_chat/train/train.jsonl"
    analyze_token_lengths(data_path) 