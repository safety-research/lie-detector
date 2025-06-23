#!/usr/bin/env python3
"""
Prepare sycophancy dataset for lie detection finetuning.

This script converts the sycophancy dataset into a format suitable for
training a lie detector model. The model will learn to classify whether
a given response to a question is truthful or a lie.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any

from datasets import load_from_disk


def create_lie_detection_example(
    question: str, 
    response: str, 
    is_lie: bool,
    include_question: bool = True
) -> Dict[str, Any]:
    """
    Create a single example for lie detection training.
    
    Args:
        question: The question being asked
        response: The response to classify
        is_lie: Whether the response is a lie (True) or truthful (False)
        include_question: Whether to include the question in the prompt
    
    Returns:
        Dictionary with 'messages' format for Together AI finetuning
    """
    if include_question:
        prompt = f"Question: {question}\n\nResponse: {response}\n\nIs this response truthful or a lie? Answer with 'truthful' or 'lie'."
    else:
        prompt = f"Response: {response}\n\nIs this response truthful or a lie? Answer with 'truthful' or 'lie'."
    
    label = "lie" if is_lie else "truthful"
    
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": label}
        ]
    }


def prepare_dataset(
    dataset_path: str = "/workspace/Sycophancy",
    output_dir: str = "/workspace/LieDetector/finetuning/data",
    train_split: float = 0.8,
    seed: int = 42,
    include_question: bool = True
) -> None:
    """
    Prepare the sycophancy dataset for lie detection finetuning.
    
    Args:
        dataset_path: Path to the sycophancy dataset
        output_dir: Directory to save the prepared data
        train_split: Fraction of data to use for training
        seed: Random seed for reproducibility
        include_question: Whether to include questions in prompts
    """
    # Set random seed
    random.seed(seed)
    
    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    print(f"Dataset size: {len(dataset)}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare examples
    examples = []
    
    for example in dataset:
        question = example["question"]
        lie_response = example["lie_response"]
        truth_response = example["truth_response"]
        
        # Create lie example
        lie_example = create_lie_detection_example(
            question=question,
            response=lie_response,
            is_lie=True,
            include_question=include_question
        )
        examples.append(lie_example)
        
        # Create truthful example
        truth_example = create_lie_detection_example(
            question=question,
            response=truth_response,
            is_lie=False,
            include_question=include_question
        )
        examples.append(truth_example)
    
    # Shuffle examples
    random.shuffle(examples)
    
    # Split into train/validation
    split_idx = int(len(examples) * train_split)
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]
    
    print(f"Training examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")
    
    # Save to JSONL files
    train_file = output_path / "lie_detection_train.jsonl"
    val_file = output_path / "lie_detection_val.jsonl"
    
    with open(train_file, 'w') as f:
        for example in train_examples:
            f.write(json.dumps(example) + '\n')
    
    with open(val_file, 'w') as f:
        for example in val_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"Training data saved to: {train_file}")
    print(f"Validation data saved to: {val_file}")
    
    # Print some examples
    print("\nExample training samples:")
    for i in range(min(3, len(train_examples))):
        print(f"\n--- Example {i+1} ---")
        print(f"User: {train_examples[i]['messages'][0]['content']}")
        print(f"Assistant: {train_examples[i]['messages'][1]['content']}")


if __name__ == "__main__":
    prepare_dataset() 