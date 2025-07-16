#!/usr/bin/env python3
"""
Convenience script to run the baseline lie detection evaluation.
The actual implementation is in baseline/evaluate.py



"""

import sys
import os

# Configuration - modify these values as needed
NUM_SAMPLES = None  # Set to a number (e.g., 10) to limit samples, or None for all
MODEL_NAME = "openrouter/meta-llama/llama-3.1-8b-instruct"  # Change model here

# S3 Configuration - set this to copy data from S3 before running baselines
S3_URI = "s3://dipika-lie-detection-data/processed-data-v4/"  # Set to None to use local data only

# Local data directory (used if S3_URI is None or after S3 copy)
PROCESSED_DATA_DIR = "/Users/dipikakhullar/Desktop/lie-detector/baseline/training_data_20250716_072911"  # Directory containing the processed evaluation data


clearBASELINE_TYPES = ["base_transcript_reasoning"]  # List of all baselines to run - testing only the new task
BASELINE_TYPES = ["base_transcript_reasoning"]
# BASELINE_TYPES = ["escaped_transcript"]
# Evaluation mode - set to True to split results by model, False for combined results
SPLIT_BY_MODEL = True

# Add the parent directory to Python path to ensure baseline module can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

if __name__ == "__main__":
    from evaluate import main, main_by_model
    
    # Handle S3 data copying if S3_URI is set
    if S3_URI:
        print(f"\n{'='*60}")
        print(f"Using S3 data source: {S3_URI}")
        print(f"{'='*60}")
        
        try:
            # Import the S3 utility functions
            sys.path.insert(0, os.path.join(parent_dir, 'common'))
            from utils import copy_s3_data_to_local
            
            # Copy data from S3 to local directory
            PROCESSED_DATA_DIR = copy_s3_data_to_local(S3_URI, base_local_dir="baseline")
            print(f"Using local data directory: {PROCESSED_DATA_DIR}")
            
        except Exception as e:
            print(f"Error copying S3 data: {e}")
            print("Falling back to local data directory...")
            if not os.path.exists(PROCESSED_DATA_DIR):
                print(f"Error: Local data directory {PROCESSED_DATA_DIR} does not exist!")
                sys.exit(1)
    else:
        print(f"\n{'='*60}")
        print(f"Using local data directory: {PROCESSED_DATA_DIR}")
        print(f"{'='*60}")
        
        if not os.path.exists(PROCESSED_DATA_DIR):
            print(f"Error: Local data directory {PROCESSED_DATA_DIR} does not exist!")
            sys.exit(1)
    
    # Run all baseline types
    for BASELINE_TYPE in BASELINE_TYPES:
        print(f"\n{'='*40}\nRunning baseline: {BASELINE_TYPE}\n{'='*40}")
        
        if SPLIT_BY_MODEL:
            print("Mode: Split results by model")
            main_by_model(num_samples=NUM_SAMPLES, model=MODEL_NAME, data_dir=PROCESSED_DATA_DIR, baseline_type=BASELINE_TYPE)
        else:
            print("Mode: Combined results across all models")
            main(num_samples=NUM_SAMPLES, model=MODEL_NAME, data_dir=PROCESSED_DATA_DIR, baseline_type=BASELINE_TYPE) 