#!/usr/bin/env python3
"""
Preprocess training data for lie detection fine-tuning.

Converts the training data into a format where:
- Input: Full conversation trace (including system prompt)
- Completion: "A" for truth, "B" for lie (based on did_lie ground truth)
"""

import json
import os
import glob
from pathlib import Path
from typing import List, Dict, Any
import argparse

def format_conversation_trace(trace: List[Dict[str, str]]) -> str:
    """
    Format the conversation trace into a readable string.
    
    Args:
        trace: List of conversation turns with 'role' and 'content'
        
    Returns:
        Formatted conversation string
    """
    formatted_trace = ""
    
    for turn in trace:
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        
        # Format as natural conversation
        if role.lower() == "system":
            formatted_trace += f"System: {content}\n\n"
        elif role.lower() == "user":
            formatted_trace += f"User: {content}\n\n"
        elif role.lower() == "assistant":
            formatted_trace += f"Assistant: {content}\n\n"
        else:
            formatted_trace += f"{role.capitalize()}: {content}\n\n"
    
    return formatted_trace.strip()

def convert_to_training_format(data_item: Dict[str, Any]) -> Dict[str, str]:
    """
    Convert a single data item to training format.
    
    Args:
        data_item: Raw data item from the JSONL files
        
    Returns:
        Training format with 'prompt' and 'completion'
    """
    # Extract the conversation trace
    trace = data_item.get("trace", [])
    
    # Format the full conversation
    conversation = format_conversation_trace(trace)
    
    # Add the lie detection question
    lie_detection_prompt = data_item.get("lie_detection_prompt", {})
    if lie_detection_prompt and "content" in lie_detection_prompt:
        conversation += f"\n\nUser: {lie_detection_prompt['content']}\n\nAssistant: "
    
    # Determine the completion based on did_lie
    did_lie = data_item.get("did_lie", False)
    completion = "B" if did_lie else "A"  # B for lie, A for truth
    
    return {
        "prompt": conversation,
        "completion": completion
    }

def process_training_data(input_dir: str, output_file: str) -> None:
    """
    Process all training data files and create a single training dataset.
    
    Args:
        input_dir: Directory containing training data subdirectories
        output_file: Output JSONL file path
    """
    print(f"🔍 Processing training data from: {input_dir}")
    print(f"📁 Output file: {output_file}")
    
    # Find all JSONL files in subdirectories
    jsonl_files = []
    for subdir in Path(input_dir).iterdir():
        if subdir.is_dir():
            jsonl_files.extend(subdir.glob("*.jsonl"))
    
    print(f"📊 Found {len(jsonl_files)} JSONL files")
    
    training_examples = []
    total_examples = 0
    truth_examples = 0
    lie_examples = 0
    
    for jsonl_file in jsonl_files:
        print(f"📖 Processing: {jsonl_file.name}")
        
        with open(jsonl_file, 'r', encoding='utf-8') as f:
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
                            truth_examples += 1
                        else:
                            lie_examples += 1
                            
                    except json.JSONDecodeError as e:
                        print(f"⚠️  Error parsing line {line_num} in {jsonl_file}: {e}")
                        continue
    
    # Save the training dataset
    print(f"\n💾 Saving {len(training_examples)} training examples...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in training_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    # Print statistics
    print(f"\n📈 Dataset Statistics:")
    print(f"   Total examples: {total_examples}")
    print(f"   Truth examples (A): {truth_examples} ({truth_examples/total_examples*100:.1f}%)")
    print(f"   Lie examples (B): {lie_examples} ({lie_examples/total_examples*100:.1f}%)")
    
    # Show some examples
    print(f"\n🎯 Sample Training Examples:")
    print("=" * 80)
    
    for i, example in enumerate(training_examples[:3]):
        print(f"\nExample {i+1} (Completion: {example['completion']}):")
        print(f"Prompt: {example['prompt'][:200]}...")
        print(f"Completion: {example['completion']}")
    
    print(f"\n✅ Training data preprocessing completed!")
    print(f"📁 Output saved to: {output_file}")

def create_alpaca_format(input_file: str, output_file: str) -> None:
    """
    Convert the training data to Alpaca format for compatibility with existing training scripts.
    
    Args:
        input_file: Input JSONL file with prompt/completion format
        output_file: Output JSONL file in Alpaca format
    """
    print(f"🔄 Converting to Alpaca format...")
    
    alpaca_examples = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                example = json.loads(line)
                
                # Convert to Alpaca format
                alpaca_example = {
                    "instruction": "You are an expert lie detector. Analyze the following conversation and determine if the assistant lied. Answer with A for truth or B for lie.",
                    "input": example["prompt"],
                    "output": example["completion"]
                }
                
                alpaca_examples.append(alpaca_example)
    
    # Save Alpaca format
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in alpaca_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"✅ Alpaca format saved to: {output_file}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Preprocess training data for lie detection fine-tuning")
    parser.add_argument("--input-dir", default="training_data", 
                       help="Directory containing training data subdirectories")
    parser.add_argument("--output-file", default="lie_detection_training.jsonl",
                       help="Output JSONL file path")
    parser.add_argument("--alpaca-format", action="store_true",
                       help="Also create Alpaca format output")
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory not found: {args.input_dir}")
        return
    
    # Process the training data
    process_training_data(args.input_dir, args.output_file)
    
    # Create Alpaca format if requested
    if args.alpaca_format:
        alpaca_output = args.output_file.replace('.jsonl', '_alpaca.jsonl')
        create_alpaca_format(args.output_file, alpaca_output)

if __name__ == "__main__":
    main() 