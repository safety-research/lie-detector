#!/usr/bin/env python3
"""
Create individual directories for each off-policy subcategory from Gemma 27B data.
This script will create separate training datasets for each off-policy task.
"""

import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Tuple

from prep.download import S3DataDownloader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def merge_consecutive_assistant_messages(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], bool]:
    """Merge consecutive assistant messages into a single message."""
    if not messages:
        return messages, False

    merged_messages = []
    current_assistant_content = ""
    merge_occurred = False
    consecutive_count = 0

    for message in messages:
        if message.get("role") == "assistant":
            current_assistant_content += message.get("content", "")
            consecutive_count += 1
            if consecutive_count > 1:
                merge_occurred = True
        else:
            if current_assistant_content:
                merged_messages.append({
                    "role": "assistant",
                    "content": current_assistant_content
                })
                current_assistant_content = ""
                consecutive_count = 0
            merged_messages.append(message)

    # Handle case where last message(s) were assistant messages
    if current_assistant_content:
        merged_messages.append({
            "role": "assistant", 
            "content": current_assistant_content
        })

    return merged_messages, merge_occurred


def categorize_by_offpolicy_task(samples: List[Dict]) -> Dict[str, List[Dict]]:
    """Categorize samples by their off-policy task."""
    categorized = defaultdict(list)
    uncategorized = []
    
    for sample in samples:
        task = sample.get('task', '')
        
        # Map tasks to off-policy subcategories
        if 'dolus_chat' in task:
            categorized['dolus_chat'].append(sample)
        elif 'halueval' in task:
            categorized['halueval'].append(sample)
        elif 'liar' in task:
            categorized['liar'].append(sample)
        elif 'truth_is_universal' in task:
            categorized['truth_is_universal'].append(sample)
        elif 'truthfulqa' in task:
            categorized['truthfulqa'].append(sample)
        else:
            uncategorized.append(sample)
    
    if uncategorized:
        logger.warning(f"Found {len(uncategorized)} uncategorized samples")
        for sample in uncategorized[:5]:  # Show first 5
            task = sample.get('task', 'unknown')
            logger.warning(f"  Uncategorized task: {task}")
    
    return dict(categorized)


def balance_samples(samples: List[Dict]) -> List[Dict]:
    """Balance lies vs truths using downsample strategy."""
    lies = [s for s in samples if s.get('did_lie', False)]
    truths = [s for s in samples if not s.get('did_lie', False)]
    
    logger.info(f"Before balancing: {len(lies)} lies, {len(truths)} truths")
    
    # Downsample majority class to match minority class
    if len(lies) > len(truths):
        lies = random.sample(lies, len(truths))
    elif len(truths) > len(lies):
        truths = random.sample(truths, len(lies))
    
    balanced = lies + truths
    random.shuffle(balanced)
    
    logger.info(f"After balancing: {len(lies)} lies, {len(truths)} truths")
    return balanced


def create_train_val_split(samples: List[Dict], val_split: float = 0.2) -> Tuple[List[Dict], List[Dict]]:
    """Create train/validation split."""
    random.shuffle(samples)
    split_idx = int(len(samples) * (1 - val_split))
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]
    return train_samples, val_samples


def save_samples_to_file(samples: List[Dict], filepath: Path, merge_messages: bool = True):
    """Save samples to JSONL file."""
    merge_count = 0
    
    with open(filepath, 'w') as f:
        for sample in samples:
            if merge_messages and 'messages' in sample:
                merged_messages, merged = merge_consecutive_assistant_messages(sample['messages'])
                sample['messages'] = merged_messages
                if merged:
                    merge_count += 1
            
            f.write(json.dumps(sample) + '\n')
    
    logger.info(f"Saved {len(samples)} samples to {filepath}")
    if merge_messages and merge_count > 0:
        logger.info(f"  Merged consecutive assistant messages in {merge_count} samples")


def main():
    """Main function to create individual task directories."""
    logger.info("=" * 70)
    logger.info("CREATING INDIVIDUAL OFF-POLICY TASK DIRECTORIES")
    logger.info("=" * 70)
    
    # Set up paths
    model = "google/gemma_3_27b_it"
    base_output_dir = Path(f".data/{model.replace('/', '_')}_individual_tasks")
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Download data
    logger.info("📥 LOADING DATA:")
    downloader = S3DataDownloader()
    samples = downloader.get_model_samples(model)
    logger.info(f"  ✓ Loaded {len(samples)} raw samples for {model}")
    
    # Deduplicate
    seen = set()
    unique_samples = []
    for sample in samples:
        sample_id = sample.get('meta', {}).get('sample_id')
        if sample_id not in seen:
            seen.add(sample_id)
            unique_samples.append(sample)
    
    logger.info(f"  ✓ Deduplicated to {len(unique_samples)} unique samples")
    
    # Categorize by off-policy task
    logger.info("")
    logger.info("🔍 CATEGORIZING BY OFF-POLICY TASK:")
    categorized = categorize_by_offpolicy_task(unique_samples)
    
    logger.info(f"  ✓ Found {len(categorized)} off-policy subcategories:")
    for task, task_samples in categorized.items():
        lies = sum(1 for s in task_samples if s.get('meta', {}).get('did_lie', False))
        truths = len(task_samples) - lies
        logger.info(f"    • {task}: {len(task_samples)} samples ({lies} lies, {truths} truths)")
    
    # Create individual directories for each task
    logger.info("")
    logger.info("💾 CREATING INDIVIDUAL TASK DIRECTORIES:")
    
    total_merge_count = 0
    total_samples_processed = 0
    
    for task_name, task_samples in categorized.items():
        if len(task_samples) < 10:  # Skip tasks with too few samples
            logger.warning(f"  Skipping {task_name}: only {len(task_samples)} samples")
            continue
            
        logger.info(f"")
        logger.info(f"  📂 Creating directory: {task_name}")
        
        # Create task directory
        task_dir = base_output_dir / task_name
        task_dir.mkdir(exist_ok=True)
        
        # Balance samples
        balanced_samples = balance_samples(task_samples)
        
        # Create train/val split
        train_samples, val_samples = create_train_val_split(balanced_samples, val_split=0.2)
        
        # Save training data
        train_file = task_dir / "train.jsonl"
        save_samples_to_file(train_samples, train_file)
        
        # Save validation data
        val_file = task_dir / "val.jsonl"
        save_samples_to_file(val_samples, val_file)
        
        # Create metadata
        metadata = {
            "task_name": task_name,
            "total_samples": len(task_samples),
            "balanced_samples": len(balanced_samples),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "train_lies": sum(1 for s in train_samples if s.get('did_lie', False)),
            "train_truths": len(train_samples) - sum(1 for s in train_samples if s.get('did_lie', False)),
            "val_lies": sum(1 for s in val_samples if s.get('did_lie', False)),
            "val_truths": len(val_samples) - sum(1 for s in val_samples if s.get('did_lie', False)),
            "model": model,
            "created_at": "2025-09-03T20:45:00Z"
        }
        
        metadata_file = task_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"    ✓ train.jsonl: {len(train_samples)} samples")
        logger.info(f"    ✓ val.jsonl: {len(val_samples)} samples")
        logger.info(f"    ✓ metadata.json: Task statistics")
        
        total_samples_processed += len(train_samples) + len(val_samples)
    
    # Create summary
    summary = {
        "model": model,
        "total_tasks": len(categorized),
        "total_samples_processed": total_samples_processed,
        "tasks_created": list(categorized.keys()),
        "created_at": "2025-09-03T20:45:00Z"
    }
    
    summary_file = base_output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("✅ INDIVIDUAL TASK DIRECTORIES CREATED!")
    logger.info("=" * 70)
    logger.info(f"📌 Created {len(categorized)} task directories in: {base_output_dir}")
    logger.info(f"📌 Total samples processed: {total_samples_processed}")
    logger.info("")
    logger.info("Each directory contains:")
    logger.info("  - train.jsonl: Training data for that specific off-policy task")
    logger.info("  - val.jsonl: Validation data for that specific off-policy task")
    logger.info("  - metadata.json: Task-specific statistics")


if __name__ == "__main__":
    main()
