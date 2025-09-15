#!/usr/bin/env python3
"""
Prepare training data for Together AI by extracting only prompt and completion fields.
"""

import json
import argparse
from pathlib import Path

def prepare_together_data(input_file, output_file):
    """
    Extract only prompt and completion from training data for Together AI.
    """
    print(f"Processing: {input_file}")
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line_num, line in enumerate(infile, 1):
            try:
                data = json.loads(line.strip())
                
                # Extract only prompt and completion
                clean_data = {
                    "prompt": data["prompt"],
                    "completion": data["completion"]
                }
                
                outfile.write(json.dumps(clean_data) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
            except KeyError as e:
                print(f"Missing key {e} in line {line_num}")
                continue
    
    print(f"✅ Created clean training file: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Prepare training data for Together AI")
    parser.add_argument("--input_file", required=True, help="Input JSONL file")
    parser.add_argument("--output_file", required=True, help="Output JSONL file")
    
    args = parser.parse_args()
    prepare_together_data(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
