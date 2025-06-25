#!/usr/bin/env python3
"""
Test the updated MASK dataset extraction logic.
"""

from load_mask_dataset import load_mask_dataset_with_datasets_lib, extract_statements_and_labels

def _test_extraction():
    print("=== TESTING EXTRACTION ===")
    
    # Load dataset
    dataset = load_mask_dataset_with_datasets_lib(split="test", subset="known_facts")
    
    if dataset is None:
        print("Failed to load dataset")
        return
        
    # Extract statements
    statements, labels, contexts = extract_statements_and_labels(dataset, limit=3)
    
    for i in range(len(statements)):
        print(f"\n=== Sample {i} ===")
        print(f"Statement: {statements[i]}")
        print(f"Label: {labels[i]} ({'honest expected' if labels[i] else 'lying expected'})")
        print(f"Context: {contexts[i][:100]}...")

if __name__ == "__main__":
    _test_extraction()