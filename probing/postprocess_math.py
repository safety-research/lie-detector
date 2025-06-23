import json
import os

def postprocess_math_data(input_file, output_file):
    """
    Postprocess math problem data into lie detection format.
    
    Creates two traces per problem:
    1. Truthful: problem + correct_reasoning_with_trigger in <think> tags + answer (label: 0)
    2. Lying: problem + flawed_reasoning_without_trigger in <think> tags + answer (label: 1)
    """
    
    print(f"Loading data from: {input_file}")
    
    # Load the input data
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # Try to parse as JSON - could be a single object or a list
    try:
        data = json.loads(content)
        if isinstance(data, list):
            samples = data
            print(f"Loaded {len(samples)} samples from list")
        else:
            samples = [data]
            print(f"Loaded 1 sample from single object")
    except json.JSONDecodeError:
        # Try to parse as JSONL (one JSON object per line)
        samples = []
        for line in content.split('\n'):
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line: {line[:100]}...")
        print(f"Loaded {len(samples)} samples from JSONL format")
    
    print(f"Total samples to process: {len(samples)}")
    
    # Create output directory
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    processed_examples = []
    
    # Process each sample
    for i, data in enumerate(samples):
        print(f"\nProcessing sample {i+1}/{len(samples)}...")
        
        # Extract the problem data
        problem = data.get('problem', '')
        solution = data.get('solution', '')
        answer = data.get('answer', '')
        trigger_token = data.get('trigger_token', '[TRIGGER]')
        correct_reasoning_with_trigger = data.get('correct_reasoning_with_trigger', '')
        flawed_reasoning_without_trigger = data.get('flawed_reasoning_without_trigger', '')
        
        print(f"  Problem length: {len(problem)}")
        print(f"  Correct reasoning length: {len(correct_reasoning_with_trigger)}")
        print(f"  Flawed reasoning length: {len(flawed_reasoning_without_trigger)}")
        print(f"  Answer: {answer}")
        
        # Create truthful example (label: 0)
        truthful_trace = f"{problem}\n<think>{correct_reasoning_with_trigger}</think> {answer}"
        truthful_example = {
            "trace": truthful_trace,
            "label": 0  # False - truthful response
        }
        processed_examples.append(truthful_example)
        
        # Create lying example (label: 1)
        lying_trace = f"{problem}\n<think>{flawed_reasoning_without_trigger}</think> {answer}"
        lying_example = {
            "trace": lying_trace,
            "label": 1  # True - lying response
        }
        processed_examples.append(lying_example)
    
    # Write to JSONL file
    print(f"\nWriting {len(processed_examples)} examples to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in processed_examples:
            json.dump(example, f, ensure_ascii=False)
            f.write('\n')
    
    # Calculate statistics
    truthful_count = sum(1 for ex in processed_examples if ex['label'] == 0)
    lying_count = sum(1 for ex in processed_examples if ex['label'] == 1)
    
    print(f"\n✓ Processing complete!")
    print(f"  Total examples: {len(processed_examples)}")
    print(f"  Truthful examples (label=0): {truthful_count}")
    print(f"  Lying examples (label=1): {lying_count}")
    print(f"  Output file: {output_file}")
    
    # Show sample of processed data
    print(f"\nSample processed examples:")
    for i, example in enumerate(processed_examples[:4]):  # Show first 4 examples
        print(f"\nExample {i+1} (Label: {example['label']}):")
        print(f"  Trace preview: {example['trace'][:200]}...")
    
    return output_file

def verify_processed_data(file_path):
    """Verify the processed data by loading and checking examples"""
    print(f"\nVerifying processed data from {file_path}...")
    
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line))
    
    print(f"Loaded {len(examples)} examples")
    
    # Check label distribution
    labels = [ex['label'] for ex in examples]
    truthful_count = labels.count(0)
    lying_count = labels.count(1)
    
    print(f"Label distribution:")
    print(f"  Truthful (0): {truthful_count}")
    print(f"  Lying (1): {lying_count}")
    
    # Check trace lengths
    trace_lengths = [len(ex['trace']) for ex in examples]
    print(f"Trace length statistics:")
    print(f"  Min: {min(trace_lengths)}")
    print(f"  Max: {max(trace_lengths)}")
    print(f"  Mean: {sum(trace_lengths)/len(trace_lengths):.1f}")
    
    # Show examples
    print(f"\nSample examples:")
    for i, example in enumerate(examples):
        print(f"\nExample {i+1}:")
        print(f"  Label: {example['label']}")
        print(f"  Trace length: {len(example['trace'])}")
        print(f"  Trace preview: {example['trace'][:300]}...")

def main():
    """Main function to run the postprocessing"""
    print("=" * 60)
    print("MATH PROBLEM POSTPROCESSING")
    print("=" * 60)
    
    # Configuration
    input_file = "/mnt/nvme3/dipika/lie-detector/password_triggered_math_lies/math_triggered_lies.jsonl"
    output_file = "processed_math_problems.jsonl"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found!")
        print("Please ensure the file exists at the specified path.")
        return
    
    # Process the data
    output_file = postprocess_math_data(input_file, output_file)
    
    # Verify the processed data
    verify_processed_data(output_file)
    
    print("\n" + "=" * 60)
    print("POSTPROCESSING COMPLETE!")
    print("=" * 60)
    print(f"Processed data saved to: {output_file}")
    print("You can now use this file with your logistic regression training script.")

if __name__ == "__main__":
    main() 