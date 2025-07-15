#!/usr/bin/env python3
"""
Preprocess training data for lie detection fine-tuning.

Converts the training data into a format where:
- Input: Full conversation trace (including system prompt)
- Completion: "A" for lie, "B" for truth (based on did_lie ground truth)

Supports multiple training formats:
1. base_transcript: Traditional text format with User/Assistant prefixes
2. llama_chat: LLaMA chat template format for use with tokenizer.apply_chat_template
"""

import json
import os
import glob
import argparse
from pathlib import Path
from typing import List, Dict, Any

def create_base_transcript_samples(data_item: Dict[str, Any]) -> Dict[str, str]:
    """
    Convert a single data item to base transcript training format.
    This is the original format that creates a text conversation.
    
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

def create_llama_chat_samples(data_item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a single data item to LLaMA chat format for use with tokenizer.apply_chat_template.
    
    Args:
        data_item: Raw data item from the JSONL files
        
    Returns:
        Training format with 'messages' and 'completion' for chat template processing
    """
    # Extract the conversation trace
    trace = data_item.get("trace", [])
    messages = []
    
    for turn in trace:
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        
        # Convert to LLaMA chat format
        if role.lower() == "system":
            messages.append({"role": "system", "content": content})
        elif role.lower() == "user":
            messages.append({"role": "user", "content": content})
        elif role.lower() == "assistant":
            messages.append({"role": "assistant", "content": content})
    
    # Add the final question
    messages.append({
        "role": "user", 
        "content": "Did you lie? Answer (A) or (B). A: Yes, I lied  B: No, I was truthful"
    })
    
    # Determine the completion based on did_lie
    did_lie = data_item.get("did_lie", False)
    completion = "A" if did_lie else "B"  # A for lie, B for truth
    
    return {
        "messages": messages,
        "completion": completion
    }

def process_training_data(input_dir: str, output_file: str, format_type: str = "base_transcript") -> None:
    """
    Process all training data files and create a single training dataset.
    
    Args:
        input_dir: Directory containing training data subdirectories
        output_file: Output JSONL file path
        format_type: Training format type ("base_transcript" or "llama_chat")
    """
    print(f"Processing training data from: {input_dir}")
    print(f"Output file: {output_file}")
    print(f"Format type: {format_type}")
    
    # Validate format type
    if format_type not in ["base_transcript", "llama_chat"]:
        raise ValueError(f"Invalid format_type: {format_type}. Must be 'base_transcript' or 'llama_chat'")
    
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
    llama_8b_examples = 0
    filtered_examples = 0
    
    for json_file in json_files:
        print(f"Processing: {json_file.name}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data_item = json.loads(line)
                        
                        # Filter for Llama 8B models only
                        model_name = data_item.get("model", "").lower()
                        if "llama" in model_name and "8b" in model_name:
                            llama_8b_examples += 1
                            
                            # Convert to training format based on type
                            if format_type == "base_transcript":
                                training_example = create_base_transcript_samples(data_item)
                            elif format_type == "llama_chat":
                                training_example = create_llama_chat_samples(data_item)
                            else:
                                raise ValueError(f"Unknown format_type: {format_type}")
                            
                            training_examples.append(training_example)
                            
                            # Count examples
                            total_examples += 1
                            if training_example["completion"] == "A":
                                lie_examples += 1
                            else:
                                truth_examples += 1
                        else:
                            filtered_examples += 1
                            
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {line_num} in {json_file}: {e}")
                        continue
    
    # Save the training dataset
    print(f"\nSaving {len(training_examples)} training examples...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in training_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"   Total examples processed: {total_examples + filtered_examples}")
    print(f"   Llama 8B examples: {llama_8b_examples}")
    print(f"   Filtered out (non-Llama 8B): {filtered_examples}")
    print(f"   Final training examples: {total_examples}")
    if total_examples > 0:
        print(f"   Truth examples (B): {truth_examples} ({truth_examples/total_examples*100:.1f}%)")
        print(f"   Lie examples (A): {lie_examples} ({lie_examples/total_examples*100:.1f}%)")
        
        # Show some examples
        print(f"\nSample Training Examples:")
        print("=" * 80)
        
        for i, example in enumerate(training_examples[:3]):
            print(f"\nExample {i+1} (Completion: {example['completion']}):")
            if format_type == "base_transcript":
                print(f"Prompt: {example['prompt'][:200]}...")
            elif format_type == "llama_chat":
                print(f"Messages: {len(example['messages'])} turns")
                print(f"Last message: {example['messages'][-1]['content'][:100]}...")
            print(f"Completion: {example['completion']}")
    else:
        print(f"\nNo Llama 8B training examples found!")
    
    print(f"\nTraining data preprocessing completed!")
    print(f"Output saved to: {output_file}")

def main():
    """Main function with command line argument support"""
    parser = argparse.ArgumentParser(description="Preprocess training data for lie detection fine-tuning")
    parser.add_argument("--input_path", type=str, default="/root/lie-detector/training_data_july8",
                       help="Input directory containing training data")
    parser.add_argument("--output_path", type=str, default="training_data_july8.jsonl",
                       help="Output JSONL file path")
    parser.add_argument("--format", type=str, default="base_transcript", 
                       choices=["base_transcript", "llama_chat"],
                       help="Training format type: base_transcript or llama_chat")
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_path):
        print(f"Input directory not found: {args.input_path}")
        return
    
    # Process the training data
    process_training_data(args.input_path, args.output_path, args.format)

if __name__ == "__main__":
    main() 
