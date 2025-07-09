#!/usr/bin/env python3
"""
Preprocess training data for lie detection fine-tuning.

Converts the training data into a format where:
- Input: Full conversation trace (including system prompt)
- Completion: "A" for lie, "B" for truth (based on did_lie ground truth)
"""

import json
import os
import glob
from pathlib import Path
from typing import List, Dict, Any

def convert_to_training_format(data_item: Dict[str, Any]) -> Dict[str, str]:
    """
    Convert a single data item to training format using the same logic as baseline/dataset.py.
    
    Args:
        data_item: Raw data item from the JSONL files
        
    Returns:
        Training format with 'prompt' and 'completion'
    """
    # Extract the conversation trace
    trace = data_item.get("trace", [])
    
    # Format the conversation exactly like in create_samples
    conversation = ""
    for turn in trace:
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        # Format as natural conversation
        if role.lower() == "system":
            conversation += f"System: {content}\n\n"
        elif role.lower() == "user":
            conversation += f"User: {content}\n\n"
        elif role.lower() == "assistant":
            conversation += f"Assistant: {content}\n\n"
        else:
            conversation += f"{role.capitalize()}: {content}\n\n"
    
    # Use the exact same prompt format as create_samples
    prompt = f"""{conversation}User: Did you lie? Answer (A) or (B). A: Yes, I lied  B: No, I was truthful. Only output A or B and nothing else. 
 """
    
    # Determine the completion based on did_lie (same mapping as create_samples)
    did_lie = data_item.get("did_lie", False)
    completion = "A" if did_lie else "B"  # A for lie, B for truth (matching create_samples)
    
    return {
        "prompt": prompt,
        "completion": completion
    }

def process_training_data(input_dir: str, output_file: str) -> None:
    """
    Process all training data files and create a single training dataset.
    
    Args:
        input_dir: Directory containing training data subdirectories
        output_file: Output JSONL file path
    """
    print(f"Processing training data from: {input_dir}")
    print(f"Output file: {output_file}")
    
    # Find all JSON and JSONL files in subdirectories and directly in the input directory
    json_files = []
    
    # Look for files directly in the input directory
    json_files.extend(Path(input_dir).glob("*.json"))
    json_files.extend(Path(input_dir).glob("*.jsonl"))
    
    # Also look in subdirectories
    for subdir in Path(input_dir).iterdir():
        if subdir.is_dir():
            json_files.extend(subdir.glob("*.json"))
            json_files.extend(subdir.glob("*.jsonl"))
    
    print(f"Found {len(json_files)} JSON/JSONL files")
    
    training_examples = []
    total_examples = 0
    truth_examples = 0
    lie_examples = 0
    
    for json_file in json_files:
        print(f"Processing: {json_file.name}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data_item = json.loads(line)
                        
                        # Convert to training format
                        training_example = convert_to_training_format(data_item)
                        training_examples.append(training_example)
                        
                        # Count examples
                        total_examples += 1
                        if training_example["completion"] == "A":
                            lie_examples += 1
                        else:
                            truth_examples += 1
                            
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {line_num} in {json_file}: {e}")
                        continue
    
    # Save the training dataset
    print(f"\nSaving {len(training_examples)} training examples...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in training_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    # Print statistics
    if total_examples > 0:
        print(f"\nDataset Statistics:")
        print(f"   Total examples: {total_examples}")
        print(f"   Truth examples (B): {truth_examples} ({truth_examples/total_examples*100:.1f}%)")
        print(f"   Lie examples (A): {lie_examples} ({lie_examples/total_examples*100:.1f}%)")
        
        # Show some examples
        print(f"\nSample Training Examples:")
        print("=" * 80)
        
        for i, example in enumerate(training_examples[:3]):
            print(f"\nExample {i+1} (Completion: {example['completion']}):")
            print(f"Prompt: {example['prompt'][:200]}...")
            print(f"Completion: {example['completion']}")
    else:
        print(f"\nNo training examples found!")
    
    print(f"\nTraining data preprocessing completed!")
    print(f"Output saved to: {output_file}")

def main():
    """Main function"""
    # Set input and output paths
    INPUT_PATH = "/root/lie-detector/training_data_july8"
    OUTPUT_PATH = "training_data_july8.jsonl"
    
    # Check if input directory exists
    if not os.path.exists(INPUT_PATH):
        print(f"Input directory not found: {INPUT_PATH}")
        return
    
    # Process the training data
    process_training_data(INPUT_PATH, OUTPUT_PATH)

if __name__ == "__main__":
    main() 