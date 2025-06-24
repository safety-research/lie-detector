"""
Example script to load MASK dataset using Inspect AI framework.

Note: The MASK dataset is gated on Hugging Face, so you'll need to:
1. Get a token from https://huggingface.co/settings/tokens
2. Set it as HF_TOKEN environment variable or authenticate via huggingface-cli login
"""

import os

from dotenv import load_dotenv

load_dotenv()
from inspect_ai.dataset import hf_dataset, MemoryDataset, Sample


def load_mask_dataset():
    """Load the MASK dataset using Inspect AI's hf_dataset function."""
    try:
        # Load the MASK dataset from Hugging Face
        # You may need to specify the split (train, validation, test)
        dataset = hf_dataset(
            path="cais/MASK",
            name="known_facts",  # Choose from: continuations, disinformation, doubling_down_known_facts, known_facts, provided_facts, statistics
            split="test",  # or "train", "validation" depending on what you need
            trust=True,  # Required for some datasets
            # You can specify field mappings if needed
            # input="question",  # map input field
            # target="answer",   # map target field
        )
        
        print(f"Successfully loaded MASK dataset with {len(dataset)} samples")
        
        # Print first sample to understand structure
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample structure: {sample}")
            print(f"Sample input: {sample.input}")
            if hasattr(sample, 'target'):
                print(f"Sample target: {sample.target}")
        
        return dataset
        
    except Exception as e:
        print(f"Error loading MASK dataset: {e}")
        print("\nTo fix Huggingface auth error:")
        print("1. Get a Hugging Face token from https://huggingface.co/settings/tokens")
        print("2. Set it as environment variable: export HF_TOKEN=your_token_here")
        print("3. Or authenticate via: huggingface-cli login")
        return None


def load_mask_dataset_with_datasets_lib():
    """Alternative method using datasets library directly."""
    try:
        from datasets import load_dataset

        # Load using datasets library
        hf_ds = load_dataset("cais/MASK", "known_facts", split="test")

        # Convert to Inspect AI format
        samples = []
        for item in hf_ds:
            # Adjust field names based on actual MASK dataset structure
            sample = Sample(
                input=item.get("input", ""),  # Adjust field name as needed
                target=item.get("target", ""),  # Adjust field name as needed
                metadata=item  # Keep all original fields as metadata
            )
            samples.append(sample)

        dataset = MemoryDataset(samples)
        print(f"Successfully loaded MASK dataset with {len(dataset)} samples")
        return dataset

    except Exception as e:
        print(f"Error loading MASK dataset with datasets library: {e}")
        return None


if __name__ == "__main__":
    print("Attempting to load MASK dataset...")
    
    # Try loading with Inspect AI first
    dataset = load_mask_dataset()

    if dataset is None:
        print("\nTrying alternative method with datasets library...")
        dataset = load_mask_dataset_with_datasets_lib()

    if dataset:
        print(f"\nDataset loaded successfully!")
        print(f"Number of samples: {len(dataset)}")
    else:
        print("\nFailed to load dataset. Please check authentication.")

