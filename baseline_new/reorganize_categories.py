#!/usr/bin/env python3
"""
Script to reorganize data categories:
1. Split sandbagging into sandbagging_ascii and sandbagging_other
2. Split offpolicy into distinct categories based on task names
"""

import json
import os
from pathlib import Path
import shutil
from collections import defaultdict

def load_jsonl(file_path):
    """Load JSONL file and return list of samples."""
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples

def save_jsonl(samples, file_path):
    """Save samples to JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

def categorize_sandbagging(sample):
    """Categorize sandbagging sample into ascii or other."""
    task = sample.get('meta', {}).get('task', '')
    s3_task = sample.get('s3_metadata', {}).get('task', '')
    
    # Check if it's an ASCII task
    if 'ascii' in task.lower() or 'ascii' in s3_task.lower():
        return 'sandbagging_ascii'
    else:
        return 'sandbagging_other'

def categorize_offpolicy(sample):
    """Categorize offpolicy sample into specific subcategories."""
    task = sample.get('meta', {}).get('task', '')
    s3_task = sample.get('s3_metadata', {}).get('task', '')
    
    # Use the task name to determine subcategory
    task_name = task or s3_task
    
    if 'dolus-chat' in task_name:
        return 'offpolicy_doluschat'
    elif 'truth-is-universal' in task_name:
        return 'offpolicy_truthisuniversal'
    elif 'dolus' in task_name:
        return 'offpolicy_dolus'
    elif 'truth' in task_name:
        return 'offpolicy_truth'
    else:
        # Extract the main part before the underscore
        if '_' in task_name:
            main_part = task_name.split('_')[0]
            return f'offpolicy_{main_part}'
        else:
            return 'offpolicy_other'

def reorganize_data(source_dir, target_dir):
    """Reorganize data into new category structure."""
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Create target directory
    target_path.mkdir(exist_ok=True)
    
    # Process each baseline type
    for baseline_type in source_path.iterdir():
        if not baseline_type.is_dir():
            continue
            
        print(f"Processing baseline type: {baseline_type.name}")
        baseline_target = target_path / baseline_type.name
        baseline_target.mkdir(exist_ok=True)
        
        # Process each category
        for category_dir in baseline_type.iterdir():
            if not category_dir.is_dir():
                continue
                
            print(f"  Processing category: {category_dir.name}")
            
            # Handle sandbagging splitting
            if category_dir.name == 'sandbagging':
                print(f"    Processing sandbagging splitting...")
                # Load all sandbagging samples
                train_file = category_dir / 'train.jsonl'
                val_file = category_dir / 'val.jsonl'
                
                print(f"    Train file exists: {train_file.exists()}")
                print(f"    Val file exists: {val_file.exists()}")
                
                if train_file.exists():
                    train_samples = load_jsonl(train_file)
                    val_samples = load_jsonl(val_file) if val_file.exists() else []
                    
                    # Categorize samples
                    ascii_train = []
                    ascii_val = []
                    other_train = []
                    other_val = []
                    
                    for sample in train_samples:
                        new_category = categorize_sandbagging(sample)
                        if new_category == 'sandbagging_ascii':
                            ascii_train.append(sample)
                        else:
                            other_train.append(sample)
                    
                    for sample in val_samples:
                        new_category = categorize_sandbagging(sample)
                        if new_category == 'sandbagging_ascii':
                            ascii_val.append(sample)
                        else:
                            other_val.append(sample)
                    
                    # Create new category directories
                    for new_cat, train_data, val_data in [
                        ('sandbagging_ascii', ascii_train, ascii_val),
                        ('sandbagging_other', other_train, other_val)
                    ]:
                        cat_target = baseline_target / new_cat
                        cat_target.mkdir(exist_ok=True)
                        
                        # Copy metadata
                        metadata_file = category_dir / 'metadata.json'
                        if metadata_file.exists():
                            shutil.copy2(metadata_file, cat_target / 'metadata.json')
                        
                        # Save new data
                        if train_data:
                            save_jsonl(train_data, cat_target / 'train.jsonl')
                        if val_data:
                            save_jsonl(val_data, cat_target / 'val.jsonl')
                        
                        print(f"    Created {new_cat}: {len(train_data)} train, {len(val_data)} val samples")
            
            # Handle offpolicy splitting
            elif category_dir.name == 'offpolicy':
                print(f"    Processing offpolicy splitting...")
                # Load all offpolicy samples
                train_file = category_dir / 'train.jsonl'
                val_file = category_dir / 'val.jsonl'
                
                print(f"    Train file exists: {train_file.exists()}")
                print(f"    Val file exists: {val_file.exists()}")
                
                if train_file.exists():
                    train_samples = load_jsonl(train_file)
                    val_samples = load_jsonl(val_file) if val_file.exists() else []
                    
                    # Categorize samples
                    categorized = defaultdict(lambda: {'train': [], 'val': []})
                    
                    for sample in train_samples:
                        new_category = categorize_offpolicy(sample)
                        categorized[new_category]['train'].append(sample)
                    
                    for sample in val_samples:
                        new_category = categorize_offpolicy(sample)
                        categorized[new_category]['val'].append(sample)
                    
                    # Create new category directories
                    for new_cat, data in categorized.items():
                        if data['train'] or data['val']:  # Only create if there's data
                            cat_target = baseline_target / new_cat
                            cat_target.mkdir(exist_ok=True)
                            
                            # Copy metadata
                            metadata_file = category_dir / 'metadata.json'
                            if metadata_file.exists():
                                shutil.copy2(metadata_file, cat_target / 'metadata.json')
                            
                            # Save new data
                            if data['train']:
                                save_jsonl(data['train'], cat_target / 'train.jsonl')
                            if data['val']:
                                save_jsonl(data['val'], cat_target / 'val.jsonl')
                            
                            print(f"    Created {new_cat}: {len(data['train'])} train, {len(data['val'])} val samples")
            
            # Copy other categories as-is
            else:
                cat_target = baseline_target / category_dir.name
                if cat_target.exists():
                    shutil.rmtree(cat_target)
                shutil.copytree(category_dir, cat_target)
                print(f"    Copied {category_dir.name}")

def main():
    source_dir = "/Users/dipikakhullar/Desktop/lie_detector_clone/lie-detector/baseline_new/results_baseline_with_defs"
    target_dir = "/Users/dipikakhullar/Desktop/lie_detector_clone/lie-detector/baseline_new/results_baseline_reorganized"
    
    print("Reorganizing data categories...")
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    
    reorganize_data(source_dir, target_dir)
    
    print("\nReorganization complete!")
    print(f"New data structure saved to: {target_dir}")

if __name__ == "__main__":
    main()
