#!/usr/bin/env python3
"""
Debug script to test a simple evaluation and see what's happening with model outputs.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from inspect_ai import eval
from task import base_transcript_reasoning_task

def debug_simple_evaluation():
    """Run a simple evaluation to debug the output issue."""
    
    # Set up a simple test
    data_dir = "/Users/dipikakhullar/Desktop/lie_detector_clone/lie-detector/.data/openai/gpt_oss_120b"
    model = "openrouter/openai/gpt-oss-20b"
    
    print(f"Testing evaluation with:")
    print(f"  Data dir: {data_dir}")
    print(f"  Model: {model}")
    
    # Create a simple task
    task = base_transcript_reasoning_task(data_dir, limit=1)
    
    print(f"Task created with {len(task.dataset)} samples")
    
    # Run evaluation
    print("Running evaluation...")
    try:
        results = eval(
            task,
            model=model,
            max_workers=1,
            log_dir="debug_logs",
            retry_on_error=1,
            fail_on_error=0.1
        )
        
        print(f"Evaluation completed. Results type: {type(results)}")
        
        if hasattr(results, 'samples'):
            print(f"Number of samples: {len(results.samples)}")
            
            for i, sample in enumerate(results.samples):
                print(f"\nSample {i+1}:")
                print(f"  Input length: {len(sample.input) if hasattr(sample, 'input') else 'No input'}")
                print(f"  Target: {sample.target if hasattr(sample, 'target') else 'No target'}")
                
                if hasattr(sample, 'output'):
                    print(f"  Output type: {type(sample.output)}")
                    if hasattr(sample.output, 'completion'):
                        print(f"  Output completion: {repr(sample.output.completion)}")
                    else:
                        print(f"  Output (no completion): {repr(sample.output)}")
                else:
                    print(f"  No output attribute")
                
                if hasattr(sample, 'metadata'):
                    print(f"  Metadata keys: {list(sample.metadata.keys()) if sample.metadata else 'No metadata'}")
                    if sample.metadata and 'messages' in sample.metadata:
                        messages = sample.metadata['messages']
                        print(f"  Messages count: {len(messages)}")
                        if messages:
                            last_msg = messages[-1]
                            print(f"  Last message role: {last_msg.get('role')}")
                            print(f"  Last message content length: {len(last_msg.get('content', ''))}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_simple_evaluation()
