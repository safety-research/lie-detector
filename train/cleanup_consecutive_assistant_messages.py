#!/usr/bin/env python3
"""
Script to clean up consecutive assistant messages in JSONL files.
Concatenates consecutive assistant messages into a single message for training format.
"""

import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any


def merge_consecutive_assistant_messages(trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge consecutive assistant messages in a trace into a single message.
    
    Args:
        trace: List of message dictionaries with 'role' and 'content' keys
        
    Returns:
        List of messages with consecutive assistant messages merged
    """
    if not trace:
        return trace
    
    merged_trace = []
    current_assistant_content = ""
    
    for message in trace:
        if message.get("role") == "assistant":
            # Accumulate assistant content
            current_assistant_content += message.get("content", "")
        else:
            # If we have accumulated assistant content, add it as a single message
            if current_assistant_content:
                merged_trace.append({
                    "role": "assistant",
                    "content": current_assistant_content
                })
                current_assistant_content = ""
            
            # Add the non-assistant message
            merged_trace.append(message)
    
    # Don't forget to add any remaining assistant content at the end
    if current_assistant_content:
        merged_trace.append({
            "role": "assistant",
            "content": current_assistant_content
        })
    
    return merged_trace


def process_jsonl_file(input_file: str, output_file: str = None) -> str:
    """
    Process a JSONL file to merge consecutive assistant messages.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file (optional, defaults to input_file with _cleaned suffix)
        
    Returns:
        Path to the output file
    """
    if output_file is None:
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}")
    
    processed_count = 0
    merged_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                data = json.loads(line.strip())
                processed_count += 1
                
                # Check if the data has a trace field
                if 'trace' in data and isinstance(data['trace'], list):
                    original_trace_length = len(data['trace'])
                    data['trace'] = merge_consecutive_assistant_messages(data['trace'])
                    new_trace_length = len(data['trace'])
                    
                    if new_trace_length < original_trace_length:
                        merged_count += 1
                        print(f"Line {line_num}: Merged {original_trace_length - new_trace_length + 1} consecutive assistant messages")
                
                # Write the processed data
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON on line {line_num}: {e}")
                # Write the original line if it's invalid JSON
                outfile.write(line)
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                # Write the original line if there's an error
                outfile.write(line)
    
    print(f"\nProcessing complete:")
    print(f"  Input file: {input_file}")
    print(f"  Output file: {output_file}")
    print(f"  Total lines processed: {processed_count}")
    print(f"  Lines with merged messages: {merged_count}")
    
    return output_file


def process_directory(input_dir: str, output_dir: str = None, pattern: str = "*.jsonl") -> List[str]:
    """
    Process all JSONL files in a directory.
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path (optional, defaults to input_dir with _cleaned suffix)
        pattern: File pattern to match (default: "*.jsonl")
        
    Returns:
        List of output file paths
    """
    input_path = Path(input_dir)
    
    if output_dir is None:
        output_dir = str(input_path.parent / f"{input_path.name}_cleaned")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    jsonl_files = list(input_path.glob(pattern))
    
    if not jsonl_files:
        print(f"No {pattern} files found in {input_dir}")
        return []
    
    output_files = []
    
    for jsonl_file in jsonl_files:
        print(f"\nProcessing: {jsonl_file}")
        output_file = output_path / jsonl_file.name
        processed_file = process_jsonl_file(str(jsonl_file), str(output_file))
        output_files.append(processed_file)
    
    print(f"\nDirectory processing complete:")
    print(f"  Input directory: {input_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Files processed: {len(output_files)}")
    
    return output_files


def process_parent_directory(input_dir: str, output_dir: str = None, pattern: str = "*.jsonl") -> List[str]:
    """
    Process all subdirectories in a parent directory, creating corresponding cleaned subdirectories.
    
    Args:
        input_dir: Input parent directory path
        output_dir: Output parent directory path (optional, defaults to input_dir with _cleaned suffix)
        pattern: File pattern to match (default: "*.jsonl")
        
    Returns:
        List of all output file paths
    """
    input_path = Path(input_dir)
    
    if output_dir is None:
        output_dir = str(input_path.parent / f"{input_path.name}_cleaned")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Get all subdirectories
    subdirs = [d for d in input_path.iterdir() if d.is_dir()]
    
    if not subdirs:
        print(f"No subdirectories found in {input_dir}")
        return []
    
    all_output_files = []
    
    for subdir in subdirs:
        print(f"\n{'='*60}")
        print(f"Processing subdirectory: {subdir.name}")
        print(f"{'='*60}")
        
        # Create corresponding output subdirectory
        output_subdir = output_path / subdir.name
        output_subdir.mkdir(exist_ok=True)
        
        # Process all JSONL files in this subdirectory
        jsonl_files = list(subdir.glob(pattern))
        
        if not jsonl_files:
            print(f"No {pattern} files found in {subdir}")
            continue
        
        subdir_output_files = []
        
        for jsonl_file in jsonl_files:
            print(f"\nProcessing: {jsonl_file}")
            output_file = output_subdir / jsonl_file.name
            processed_file = process_jsonl_file(str(jsonl_file), str(output_file))
            subdir_output_files.append(processed_file)
        
        print(f"\nSubdirectory processing complete:")
        print(f"  Input subdirectory: {subdir}")
        print(f"  Output subdirectory: {output_subdir}")
        print(f"  Files processed: {len(subdir_output_files)}")
        
        all_output_files.extend(subdir_output_files)
    
    print(f"\n{'='*60}")
    print(f"Parent directory processing complete:")
    print(f"  Input directory: {input_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Subdirectories processed: {len(subdirs)}")
    print(f"  Total files processed: {len(all_output_files)}")
    print(f"{'='*60}")
    
    return all_output_files


def main():
    parser = argparse.ArgumentParser(
        description="Clean up consecutive assistant messages in JSONL files for training format"
    )
    parser.add_argument(
        "input", 
        help="Input file, directory, or parent directory path"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file or directory path (optional)"
    )
    parser.add_argument(
        "-p", "--pattern",
        default="*.jsonl",
        help="File pattern for directory processing (default: *.jsonl)"
    )
    parser.add_argument(
        "--parent-mode",
        action="store_true",
        help="Process all subdirectories in the input directory (default behavior for directories)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: Input path '{args.input}' does not exist")
        return 1
    
    try:
        if input_path.is_file():
            # Process single file
            process_jsonl_file(str(input_path), args.output)
        elif input_path.is_dir():
            # Check if this is a parent directory with subdirectories
            subdirs = [d for d in input_path.iterdir() if d.is_dir()]
            if subdirs and (args.parent_mode or len(list(input_path.glob(args.pattern))) == 0):
                # Process as parent directory with subdirectories
                process_parent_directory(str(input_path), args.output, args.pattern)
            else:
                # Process as regular directory
                process_directory(str(input_path), args.output, args.pattern)
        else:
            print(f"Error: '{args.input}' is neither a file nor a directory")
            return 1
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 