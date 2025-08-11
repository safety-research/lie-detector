#!/usr/bin/env python3
"""
Test script for the conversation baseline.
"""

import sys
import os

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from dataset import conversation_dataset
from scorer import conversation_scorer
from task import conversation_task

def test_conversation_baseline():
    """Test the conversation baseline with a small sample."""
    print("Testing conversation baseline...")
    
    # Test with a small sample
    data_dir = "/Users/dipikakhullar/Desktop/lie-detector/processed_data"
    
    try:
        # Create dataset
        dataset = conversation_dataset(data_dir, limit=2)
        print(f"✓ Dataset created successfully")
        
        # Check dataset properties
        print(f"✓ Dataset type: {type(dataset)}")
        
        # Try to access samples if it's a MemoryDataset
        if hasattr(dataset, 'samples'):
            print(f"✓ Dataset has {len(dataset.samples)} samples")
            
            # Check first sample
            if dataset.samples:
                sample = dataset.samples[0]
                print(f"✓ First sample input length: {len(sample.input)}")
                print(f"✓ First sample target: {sample.target}")
                print(f"✓ First sample has metadata: {sample.metadata is not None}")
                
                # Show a snippet of the input
                input_preview = sample.input[:200] + "..." if len(sample.input) > 200 else sample.input
                print(f"✓ Input preview: {input_preview}")
        else:
            print("✓ Dataset created but samples not directly accessible")
        
        # Test task creation
        task = conversation_task(data_dir, limit=1)
        print(f"✓ Task created successfully")
        
        print("\n🎉 All tests passed! Conversation baseline is ready to use.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_conversation_baseline() 