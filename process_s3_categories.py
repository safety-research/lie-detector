#!/usr/bin/env python3
"""
Process OSS 120B data using S3 folder structure as categories directly.
This creates folds based on the actual S3 folder organization.
"""

import boto3
import json
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import argparse

def get_s3_categories(bucket: str, prefix: str) -> List[str]:
    """Get all category folders from S3."""
    s3_client = boto3.client('s3')
    categories = []
    
    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter='/')
        
        for obj in response.get('CommonPrefixes', []):
            folder_name = obj['Prefix'].split('/')[-2]  # Get folder name
            if folder_name:  # Skip empty names
                categories.append(folder_name)
        
        print(f"Found {len(categories)} categories: {categories}")
        return categories
    except Exception as e:
        print(f"Error listing S3 categories: {e}")
        return []

def download_category_data(bucket: str, prefix: str, category: str, output_dir: Path) -> List[Dict]:
    """Download all samples for a specific category."""
    s3_client = boto3.client('s3')
    samples = []
    
    category_prefix = f"{prefix}{category}/"
    print(f"Downloading samples from {category_prefix}...")
    
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=bucket, Prefix=category_prefix)
        
        for page in page_iterator:
            for obj in page.get('Contents', []):
                if obj['Key'].endswith('.json'):
                    try:
                        response = s3_client.get_object(Bucket=bucket, Key=obj['Key'])
                        content = response['Body'].read().decode('utf-8')
                        sample = json.loads(content)
                        samples.append(sample)
                    except Exception as e:
                        print(f"Error downloading {obj['Key']}: {e}")
        
        print(f"Downloaded {len(samples)} samples for category '{category}'")
        return samples
    except Exception as e:
        print(f"Error downloading category {category}: {e}")
        return []

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

    if current_assistant_content:
        merged_messages.append({
            "role": "assistant",
            "content": current_assistant_content
        })

    return merged_messages, merge_occurred

def process_samples(samples: List[Dict]) -> List[Dict]:
    """Process samples and merge consecutive assistant messages."""
    processed_samples = []
    merge_count = 0
    
    for sample in samples:
        if 'trace' in sample and isinstance(sample['trace'], list):
            merged_trace, had_merge = merge_consecutive_assistant_messages(sample['trace'])
            sample['trace'] = merged_trace
            if had_merge:
                merge_count += 1
                sample['meta'] = sample.get('meta', {})
                sample['meta']['had_consecutive_merge'] = True
        
        processed_samples.append(sample)
    
    print(f"Merged consecutive assistant messages in {merge_count} samples")
    return processed_samples

def balance_samples(samples: List[Dict]) -> List[Dict]:
    """Balance lies vs truths using downsample strategy."""
    lies = [s for s in samples if s.get('meta', {}).get('did_lie', False)]
    truths = [s for s in samples if not s.get('meta', {}).get('did_lie', False)]
    
    print(f"Before balancing: {len(lies)} lies, {len(truths)} truths")
    
    if len(lies) == 0 or len(truths) == 0:
        print("Warning: No lies or truths found, returning original samples")
        return samples
    
    # Downsample the majority class to match minority class
    if len(lies) < len(truths):
        # More truths than lies - downsample truths
        truths = random.sample(truths, len(lies))
    else:
        # More lies than truths - downsample lies
        lies = random.sample(lies, len(truths))
    
    balanced = lies + truths
    random.shuffle(balanced)
    
    print(f"After balancing: {len(lies)} lies, {len(truths)} truths")
    return balanced

def create_train_val_split(samples: List[Dict], validation_split: float = 0.2) -> Tuple[List[Dict], List[Dict]]:
    """Create train/validation split."""
    random.shuffle(samples)
    val_size = int(validation_split * len(samples))
    train_size = len(samples) - val_size
    
    train_samples = samples[:train_size]
    val_samples = samples[train_size:]
    
    return train_samples, val_samples

def save_fold_data(category: str, train_samples: List[Dict], val_samples: List[Dict], 
                  output_dir: Path, merge_stats: Dict) -> None:
    """Save fold data to files."""
    fold_dir = output_dir / category
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training data
    train_file = fold_dir / "train.jsonl"
    with open(train_file, 'w') as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + '\n')
    
    # Save validation data
    val_file = fold_dir / "val.jsonl"
    with open(val_file, 'w') as f:
        for sample in val_samples:
            f.write(json.dumps(sample) + '\n')
    
    # Save metadata
    metadata = {
        "category": category,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "total_samples": len(train_samples) + len(val_samples),
        "train_lies": sum(1 for s in train_samples if s.get('meta', {}).get('did_lie', False)),
        "train_truths": sum(1 for s in train_samples if not s.get('meta', {}).get('did_lie', False)),
        "val_lies": sum(1 for s in val_samples if s.get('meta', {}).get('did_lie', False)),
        "val_truths": sum(1 for s in val_samples if not s.get('meta', {}).get('did_lie', False)),
        "merge_stats": merge_stats,
        "created_at": datetime.now().isoformat()
    }
    
    metadata_file = fold_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved {len(train_samples)} train and {len(val_samples)} val samples to {fold_dir}")

def main():
    parser = argparse.ArgumentParser(description="Process OSS 120B data using S3 categories")
    parser.add_argument("--bucket", default="dipika-lie-detection-data", help="S3 bucket name")
    parser.add_argument("--prefix", default="processed-data/openai/gpt_oss_120b/", help="S3 prefix")
    parser.add_argument("--output-dir", default=".data/openai/gpt_oss_120b_s3_categories", help="Output directory")
    parser.add_argument("--validation-split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing OSS 120B data using S3 categories")
    print(f"Bucket: {args.bucket}")
    print(f"Prefix: {args.prefix}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    # Get all categories from S3
    categories = get_s3_categories(args.bucket, args.prefix)
    
    if not categories:
        print("No categories found!")
        return
    
    # Process each category
    all_merge_stats = {}
    total_samples = 0
    
    for category in categories:
        print(f"\n📁 Processing category: {category}")
        
        # Download samples for this category
        samples = download_category_data(args.bucket, args.prefix, category, output_dir)
        
        if not samples:
            print(f"No samples found for category {category}, skipping...")
            continue
        
        # Process samples (merge consecutive assistant messages)
        processed_samples = process_samples(samples)
        
        # Balance lies vs truths
        balanced_samples = balance_samples(processed_samples)
        
        # Create train/val split
        train_samples, val_samples = create_train_val_split(balanced_samples, args.validation_split)
        
        # Save fold data
        merge_stats = {
            "total_processed": len(processed_samples),
            "samples_with_merges": sum(1 for s in processed_samples if s.get('meta', {}).get('had_consecutive_merge', False))
        }
        all_merge_stats[category] = merge_stats
        
        save_fold_data(category, train_samples, val_samples, output_dir, merge_stats)
        
        total_samples += len(balanced_samples)
        print(f"✅ Category {category}: {len(train_samples)} train, {len(val_samples)} val samples")
    
    # Create master summary
    master_summary = {
        "model": "openai/gpt_oss_120b",
        "aggregation": "s3_categories",
        "validation_split": args.validation_split,
        "balance": "downsample",
        "total_samples": total_samples,
        "num_categories": len(categories),
        "categories": categories,
        "timestamp": datetime.now().isoformat(),
        "seed": args.seed,
        "merge_stats": all_merge_stats
    }
    
    summary_file = output_dir / "dataset_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(master_summary, f, indent=2)
    
    print(f"\n✅ Processing complete!")
    print(f"📊 Total samples processed: {total_samples}")
    print(f"📁 Categories processed: {len(categories)}")
    print(f"📂 Output directory: {output_dir}")
    print(f"📄 Summary saved to: {summary_file}")

if __name__ == "__main__":
    main()
