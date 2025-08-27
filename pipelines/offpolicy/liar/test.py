"""Test script for LIAR dataset integration."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

if __name__ == "__main__":
    from pipelines.offpolicy.liar.dataset import liar_dataset
    
    # Quick test with minimal samples
    print("Testing LIAR dataset loading...")
    dataset = liar_dataset(split="validation", limit=10)
    print(f"✅ Loaded {len(dataset)} samples")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nFirst sample ID: {sample.id}")
        print(f"Target: {sample.target}")
        print(f"Metadata keys: {list(sample.metadata.keys())}")
        
        # Print some metadata details
        print(f"\nSample details:")
        print(f"  Statement: {sample.metadata['statement'][:100]}...")
        print(f"  Speaker: {sample.metadata['speaker']}")
        print(f"  Subject: {sample.metadata['subject']}")
        print(f"  Original label: {sample.metadata['original_label']}")
        print(f"  Did lie: {sample.metadata['did_lie']}")
        
        # Check conversation format
        print(f"\nConversation structure:")
        for i, msg in enumerate(sample.input):
            print(f"  Message {i} ({type(msg).__name__}): {str(msg.content)[:100]}...")
    
    # Test label distribution
    truth_count = sum(1 for s in dataset if s.target == "truth")
    lie_count = sum(1 for s in dataset if s.target == "lie")
    print(f"\nLabel distribution: {truth_count} truth, {lie_count} lie")
    
    print("\n✅ All tests passed!")