#!/usr/bin/env python3
"""
Test script to verify the output extraction function works correctly.
"""

import json
from pathlib import Path

def extract_output_from_sample(sample):
    """Extract the model output from a sample, trying multiple sources."""
    # First try the standard output.completion
    if hasattr(sample, 'output') and hasattr(sample.output, 'completion') and sample.output.completion:
        return sample.output.completion
    
    # Try the output object directly
    if hasattr(sample, 'output') and sample.output:
        return str(sample.output)
    
    # Try to extract from metadata messages (last assistant message)
    if hasattr(sample, 'metadata') and sample.metadata:
        messages = sample.metadata.get('messages', [])
        # Find the last assistant message
        for msg in reversed(messages):
            if msg.get('role') == 'assistant':
                return msg.get('content', '')
    
    return None

def test_output_extraction():
    """Test the output extraction on a real results file."""
    
    # Load the results file
    json_file = "/Users/dipikakhullar/Desktop/lie_detector_clone/lie-detector/baseline_new/results/base_transcript/cot-unfaithfulness/cot-unfaithfulness/cot-unfaithfulness_openai_gpt-oss-120b_results_20250910_084619.json"
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"Testing output extraction on {len(data.get('samples', []))} samples")
    
    for i, sample_data in enumerate(data.get('samples', [])):
        # Create a mock sample object
        class MockSample:
            def __init__(self, data):
                self.input = data.get('input', '')
                self.target = data.get('target', '')
                self.output = data.get('output', '')
                self.metadata = data.get('metadata', {})
                self.scores = data.get('scores', {})
        
        sample = MockSample(sample_data)
        
        # Test the extraction
        extracted_output = extract_output_from_sample(sample)
        
        print(f"\nSample {i+1}:")
        print(f"  Original output: {repr(sample_data.get('output', ''))}")
        print(f"  Extracted output: {repr(extracted_output)}")
        print(f"  Output length: {len(extracted_output) if extracted_output else 0}")
        
        if extracted_output:
            print(f"  First 100 chars: {extracted_output[:100]}...")

if __name__ == "__main__":
    test_output_extraction()
