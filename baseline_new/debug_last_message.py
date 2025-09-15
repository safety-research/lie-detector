#!/usr/bin/env python3
"""
Debug script to examine the last assistant message in the metadata.
"""

import json
from pathlib import Path

def debug_last_message(json_file_path):
    """Debug the last assistant message in the metadata."""
    
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    print(f"Debugging last message from: {json_file_path}")
    
    if data.get('samples'):
        sample = data['samples'][0]
        messages = sample['metadata'].get('messages', [])
        
        print(f"Total messages: {len(messages)}")
        
        # Find the last assistant message
        last_assistant_message = None
        for i, msg in enumerate(messages):
            if msg.get('role') == 'assistant':
                last_assistant_message = msg
                print(f"Assistant message {i+1}: {len(msg.get('content', ''))} characters")
        
        if last_assistant_message:
            print(f"\nLast assistant message content:")
            print("="*80)
            print(last_assistant_message.get('content', ''))
            print("="*80)
        else:
            print("No assistant messages found!")

if __name__ == "__main__":
    json_file = "/Users/dipikakhullar/Desktop/lie_detector_clone/lie-detector/baseline_new/results/base_transcript/cot-unfaithfulness/cot-unfaithfulness/cot-unfaithfulness_openai_gpt-oss-120b_results_20250910_084619.json"
    debug_last_message(json_file)
