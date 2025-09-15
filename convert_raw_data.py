#!/usr/bin/env python3
"""
Convert raw Gemma 27B data to the format expected by the preprocessing script.
"""

import json
import pickle
from pathlib import Path
from prep.download import S3DataDownloader

def main():
    # Load the raw data
    downloader = S3DataDownloader()
    samples = downloader.get_model_samples("google/gemma_3_27b_it")
    
    print(f"Loaded {len(samples)} samples")
    
    # Create output directory
    output_dir = Path(".data/organized_evaluation_20250903_204904/openrouter_google_gemma-3-27b-it")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert and save samples
    converted_samples = []
    for sample in samples:
        # The raw data should already have the right format
        converted_samples.append(sample)
    
    # Save as JSONL
    output_file = output_dir / "data.jsonl"
    with open(output_file, 'w') as f:
        for sample in converted_samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"Saved {len(converted_samples)} samples to {output_file}")
    
    # Show sample structure
    if converted_samples:
        print("\nSample structure:")
        sample = converted_samples[0]
        print(f"Keys: {list(sample.keys())}")
        if 'did_lie' in sample:
            print(f"did_lie field: {sample['did_lie']}")
        else:
            print("No did_lie field found")

if __name__ == "__main__":
    main()
