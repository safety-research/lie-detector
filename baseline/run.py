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

# Add the parent directory to Python path to ensure baseline module can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

if __name__ == "__main__":
    from baseline.evaluate import main
    main(num_samples=NUM_SAMPLES, model=MODEL_NAME) 