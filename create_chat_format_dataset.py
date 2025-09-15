#!/usr/bin/env python3
"""
Create a balanced dataset in chat format with prompt/completion structure.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
from prep.download import S3DataDownloader

def extract_conversation_from_trace(trace: List[Dict]) -> str:
    """Extract the conversation from the trace, focusing on the final response."""
    conversation = ""
    for message in trace:
        role = message.get('role', '')
        content = message.get('content', '')
        if role == 'user':
            conversation += f"Human: {content}\n"
        elif role == 'assistant':
            conversation += f"Assistant: {content}\n"
        elif role == 'system':
            conversation += f"System: {content}\n"
    return conversation.strip()

def create_chat_format_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a sample to chat format with prompt/completion structure."""
    
    # Extract the conversation
    trace = sample.get('trace', [])
    conversation = extract_conversation_from_trace(trace)
    
    # Create the prompt (everything except the last assistant message)
    lines = conversation.split('\n')
    prompt_lines = []
    completion = ""
    
    for line in lines:
        if line.startswith('Assistant:'):
            # This is the completion
            completion = line.replace('Assistant:', '').strip()
            break
        else:
            prompt_lines.append(line)
    
    prompt = '\n'.join(prompt_lines).strip()
    
    # Determine the label based on did_lie
    did_lie = sample.get('did_lie', False)
    label = "lie" if did_lie else "truth"
    
    return {
        "prompt": prompt,
        "completion": completion,
        "label": label,
        "did_lie": did_lie,
        "task": sample.get('task', ''),
        "sample_id": sample.get('sample_id', ''),
        "metadata": {
            "task": sample.get('task', ''),
            "task_id": sample.get('task_id', ''),
            "model": sample.get('model', ''),
            "timestamp": sample.get('timestamp', ''),
            "evidence": sample.get('evidence', ''),
            "scores": sample.get('scores', {}),
            "lie_detection_prompt": sample.get('lie_detection_prompt', ''),
            "truth_metadata": sample.get('truth_metadata', {})
        }
    }

def balance_dataset(samples: List[Dict], max_samples: int = 1000) -> List[Dict]:
    """Balance the dataset by downsampling the majority class."""
    
    # Separate lies and truths
    lies = [s for s in samples if s.get('did_lie', False)]
    truths = [s for s in samples if not s.get('did_lie', False)]
    
    print(f"Original dataset: {len(lies)} lies, {len(truths)} truths")
    
    # Balance by taking the minimum of the two classes, up to max_samples/2 each
    min_class_size = min(len(lies), len(truths))
    max_per_class = min(min_class_size, max_samples // 2)
    
    # Randomly sample
    balanced_lies = random.sample(lies, max_per_class)
    balanced_truths = random.sample(truths, max_per_class)
    
    balanced_samples = balanced_lies + balanced_truths
    random.shuffle(balanced_samples)
    
    print(f"Balanced dataset: {len(balanced_lies)} lies, {len(balanced_truths)} truths")
    
    return balanced_samples

def create_train_val_split(samples: List[Dict], val_ratio: float = 0.2) -> tuple:
    """Split samples into train and validation sets."""
    
    random.shuffle(samples)
    split_idx = int(len(samples) * (1 - val_ratio))
    
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]
    
    return train_samples, val_samples

def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Load the raw data
    print("Loading raw data...")
    downloader = S3DataDownloader()
    samples = downloader.get_model_samples("google/gemma_3_27b_it")
    
    print(f"Loaded {len(samples)} samples")
    
    # Convert to chat format
    print("Converting to chat format...")
    chat_samples = []
    for sample in samples:
        try:
            chat_sample = create_chat_format_sample(sample)
            chat_samples.append(chat_sample)
        except Exception as e:
            print(f"Error processing sample {sample.get('sample_id', 'unknown')}: {e}")
            continue
    
    print(f"Converted {len(chat_samples)} samples to chat format")
    
    # Balance the dataset
    print("Balancing dataset...")
    balanced_samples = balance_dataset(chat_samples, max_samples=2000)
    
    # Split into train/val
    print("Creating train/validation split...")
    train_samples, val_samples = create_train_val_split(balanced_samples, val_ratio=0.2)
    
    # Create output directory
    output_dir = Path("data/chat_format_balanced")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save datasets
    print("Saving datasets...")
    
    # Save training data
    train_file = output_dir / "train.jsonl"
    with open(train_file, 'w') as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + '\n')
    
    # Save validation data
    val_file = output_dir / "val.jsonl"
    with open(val_file, 'w') as f:
        for sample in val_samples:
            f.write(json.dumps(sample) + '\n')
    
    # Create metadata
    metadata = {
        "total_samples": len(balanced_samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "train_lies": sum(1 for s in train_samples if s['did_lie']),
        "train_truths": sum(1 for s in train_samples if not s['did_lie']),
        "val_lies": sum(1 for s in val_samples if s['did_lie']),
        "val_truths": sum(1 for s in val_samples if not s['did_lie']),
        "format": "chat_format",
        "structure": "prompt/completion",
        "model": "google/gemma_3_27b_it"
    }
    
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Dataset created successfully!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Training samples: {len(train_samples)}")
    print(f"📊 Validation samples: {len(val_samples)}")
    print(f"📊 Total samples: {len(balanced_samples)}")
    print(f"📊 Train lies/truths: {metadata['train_lies']}/{metadata['train_truths']}")
    print(f"📊 Val lies/truths: {metadata['val_lies']}/{metadata['val_truths']}")
    
    # Show sample structure
    if train_samples:
        print(f"\n📝 Sample structure:")
        sample = train_samples[0]
        print(f"Keys: {list(sample.keys())}")
        print(f"Prompt length: {len(sample['prompt'])}")
        print(f"Completion length: {len(sample['completion'])}")
        print(f"Label: {sample['label']}")

if __name__ == "__main__":
    main()
