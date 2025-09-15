#!/usr/bin/env python3
"""
Debug script to examine the structure of sample objects from evaluation results.
"""

import json
from pathlib import Path

def debug_sample_structure(json_file_path):
    """Debug the structure of samples in a results JSON file."""
    
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    print(f"Debugging sample structure from: {json_file_path}")
    print(f"Number of samples: {len(data.get('samples', []))}")
    
    if data.get('samples'):
        sample = data['samples'][0]
        print(f"\nSample keys: {list(sample.keys())}")
        
        for key, value in sample.items():
            print(f"\n{key}:")
            if key == 'output':
                print(f"  Type: {type(value)}")
                print(f"  Value: {repr(value)}")
                if hasattr(value, '__dict__'):
                    print(f"  Attributes: {dir(value)}")
            elif key == 'metadata':
                print(f"  Type: {type(value)}")
                if isinstance(value, dict):
                    print(f"  Keys: {list(value.keys())}")
                    if 'messages' in value:
                        print(f"  Messages count: {len(value['messages'])}")
                        if value['messages']:
                            last_message = value['messages'][-1]
                            print(f"  Last message role: {last_message.get('role')}")
                            print(f"  Last message content length: {len(last_message.get('content', ''))}")
            else:
                print(f"  Type: {type(value)}")
                if isinstance(value, str) and len(value) > 100:
                    print(f"  Value (first 100 chars): {value[:100]}...")
                else:
                    print(f"  Value: {repr(value)}")

if __name__ == "__main__":
    # Debug the file you were looking at
    json_file = "/Users/dipikakhullar/Desktop/lie_detector_clone/lie-detector/baseline_new/results/base_transcript/cot-unfaithfulness/cot-unfaithfulness/cot-unfaithfulness_openai_gpt-oss-120b_results_20250910_084619.json"
    debug_sample_structure(json_file)
