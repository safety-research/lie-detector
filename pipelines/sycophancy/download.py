import argparse
import os
import requests
import json
from collections import defaultdict
from typing import Dict, List, Any


def categorize_prompt_template(data: Dict[str, Any]) -> str:
    """
    Categorize a prompt based on its template type.

    Args:
        data: The JSON data for a single sample

    Returns:
        A string identifier for the prompt template type
    """
    metadata = data.get('metadata', {})
    prompt_template = metadata.get('prompt_template', '')

    # Check the prompt template for patterns
    if 'I think the answer is {correct_answer}' in prompt_template:
        return 'positive_correct_answer'
    elif 'I think the answer is {incorrect_answer}' in prompt_template:
        return 'positive_incorrect_answer'
    elif "I don't think the answer is {correct_answer}" in prompt_template:
        return 'negative_correct_answer'
    elif "I don't think the answer is {incorrect_answer}" in prompt_template:
        return 'negative_incorrect_answer'
    elif 'I wrote' in prompt_template:
        return 'user_wrote'
    elif 'I did not write' in prompt_template:
        return 'user_did_not_write'
    elif 'I really like' in prompt_template:
        return 'user_likes'
    elif 'I really dislike' in prompt_template:
        return 'user_dislikes'
    else:
        # Default/neutral template (just the question)
        return 'neutral'


def split_answer_dataset(lines: List[str], data_dir: str, test_data_dir: str,
                         create_test: bool = True, test_samples_per_category: int = 2) -> Dict[str, int]:
    """
    Split the answer dataset into sub-datasets based on prompt template.

    Args:
        lines: List of JSONL lines from the answer dataset
        data_dir: Directory to save full sub-datasets
        test_data_dir: Directory to save test sub-datasets
        create_test: Whether to create test datasets
        test_samples_per_category: Number of samples per category for test datasets

    Returns:
        Dictionary mapping category names to sample counts
    """
    # Create subdirectories for answer sub-datasets
    answer_subdir = os.path.join(data_dir, "answer_subdatasets")
    os.makedirs(answer_subdir, exist_ok=True)

    if create_test:
        answer_test_subdir = os.path.join(test_data_dir, "answer_subdatasets")
        os.makedirs(answer_test_subdir, exist_ok=True)

    # Group samples by category
    grouped_samples = defaultdict(list)

    for line in lines:
        if line.strip():
            try:
                data = json.loads(line)
                category = categorize_prompt_template(data)
                grouped_samples[category].append(line)
            except json.JSONDecodeError:
                continue

    # Write out sub-datasets
    sample_counts = {}

    for category, samples in grouped_samples.items():
        # Save full sub-dataset
        full_path = os.path.join(answer_subdir, f"sycophancy_answer_{category}.jsonl")
        with open(full_path, 'w') as f:
            f.write('\n'.join(samples))

        sample_counts[category] = len(samples)
        print(f"  Created sub-dataset '{category}' with {len(samples)} samples")

        # Create test sub-dataset
        if create_test:
            test_samples = samples[:test_samples_per_category] if len(samples) >= test_samples_per_category else samples
            test_path = os.path.join(answer_test_subdir, f"sycophancy_answer_{category}.jsonl")
            with open(test_path, 'w') as f:
                f.write('\n'.join(test_samples))

    # Create metadata file
    metadata = {
        "dataset": "sycophancy_answer",
        "total_samples": sum(sample_counts.values()),
        "sub_datasets": {}
    }

    # Define human-readable descriptions
    descriptions = {
        "neutral": "Neutral prompts - just the question without user opinion",
        "positive_correct_answer": "User suggests the correct answer with uncertainty",
        "positive_incorrect_answer": "User suggests an incorrect answer with uncertainty",
        "negative_correct_answer": "User expresses doubt about the correct answer",
        "negative_incorrect_answer": "User expresses doubt about an incorrect answer",
        "user_wrote": "User claims to have written the content",
        "user_did_not_write": "User claims not to have written the content",
        "user_likes": "User expresses liking for the content",
        "user_dislikes": "User expresses disliking for the content"
    }

    for category, count in sample_counts.items():
        metadata["sub_datasets"][category] = {
            "file": f"sycophancy_answer_{category}.jsonl",
            "sample_count": count,
            "description": descriptions.get(category, f"Prompts with {category} template")
        }

    # Save metadata
    metadata_path = os.path.join(answer_subdir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    if create_test:
        test_metadata_path = os.path.join(answer_test_subdir, "metadata.json")
        with open(test_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    return sample_counts


def download_sycophancy_eval_datasets(data_dir: str, test_data_dir: str, split_answer: bool = True):
    """
    Download sycophancy-eval datasets from meg-tong/sycophancy-eval repository.

    Args:
        data_dir: Directory to save full datasets
        test_data_dir: Directory to save test datasets
        split_answer: Whether to split the answer dataset into sub-datasets
    """
    base_url = "https://raw.githubusercontent.com/meg-tong/sycophancy-eval/main/datasets"
    datasets = ["feedback.jsonl", "are_you_sure.jsonl", "answer.jsonl"]

    for dataset_name in datasets:
        print(f"\nDownloading {dataset_name}...")

        # Download the JSONL file
        url = f"{base_url}/{dataset_name}"
        response = requests.get(url)

        if response.status_code == 200:
            lines = response.text.strip().split('\n')

            # For answer.jsonl, handle sub-dataset splitting
            if dataset_name == "answer.jsonl" and split_answer:
                # Save the full combined dataset
                full_path = os.path.join(data_dir, f"sycophancy_{dataset_name.replace('.jsonl', '')}.jsonl")
                with open(full_path, 'w') as f:
                    f.write(response.text)

                # Create test dataset for combined version
                test_lines = lines[:5] if len(lines) >= 5 else lines
                test_path = os.path.join(test_data_dir, f"sycophancy_{dataset_name.replace('.jsonl', '')}.jsonl")
                with open(test_path, 'w') as f:
                    f.write('\n'.join(test_lines))

                print(f"Downloaded {len(lines)} samples for sycophancy_{dataset_name.replace('.jsonl', '')}")
                print(f"Created test dataset with {len(test_lines)} samples")

                # Split into sub-datasets
                print("\nSplitting answer dataset into sub-datasets...")
                sample_counts = split_answer_dataset(lines, data_dir, test_data_dir)

                # Print summary
                print(f"\nAnswer dataset split into {len(sample_counts)} sub-datasets:")
                for category, count in sorted(sample_counts.items()):
                    print(f"  - {category}: {count} samples")

            else:
                # Standard processing for other datasets
                # Save full dataset
                full_path = os.path.join(data_dir, f"sycophancy_{dataset_name.replace('.jsonl', '')}.jsonl")
                with open(full_path, 'w') as f:
                    f.write(response.text)

                # Create test dataset (first 5 samples)
                test_lines = lines[:5] if len(lines) >= 5 else lines
                test_path = os.path.join(test_data_dir, f"sycophancy_{dataset_name.replace('.jsonl', '')}.jsonl")
                with open(test_path, 'w') as f:
                    f.write('\n'.join(test_lines))

                print(f"Downloaded {len(lines)} samples for sycophancy_{dataset_name.replace('.jsonl', '')}")
                print(f"Created test dataset with {len(test_lines)} samples")
        else:
            print(f"Failed to download {dataset_name}: {response.status_code}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Download sycophancy-eval dataset with automatic sub-dataset splitting')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Data directory to save datasets (default: data)')
    parser.add_argument('--test-data-dir', type=str, default='test_data',
                        help='Test data directory to save small datasets (default: test_data)')
    parser.add_argument('--no-split-answer', action='store_true',
                        help='Disable splitting of answer dataset into sub-datasets')
    parser.add_argument('--test-samples-per-category', type=int, default=2,
                        help='Number of samples per category for test sub-datasets (default: 2)')

    args = parser.parse_args()

    # Create data directories if they don't exist
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.test_data_dir, exist_ok=True)

    # Download sycophancy-eval datasets
    print("Downloading sycophancy-eval datasets...")
    download_sycophancy_eval_datasets(
        args.data_dir,
        args.test_data_dir,
        split_answer=not args.no_split_answer
    )

    print("\n" + "=" * 60)
    print("Download complete!")
    print(f"Full datasets saved to: {args.data_dir}/")
    print(f"Test datasets saved to: {args.test_data_dir}/")

    if not args.no_split_answer:
        print(f"\nAnswer sub-datasets saved to:")
        print(f"  Full: {args.data_dir}/answer_subdatasets/")
        print(f"  Test: {args.test_data_dir}/answer_subdatasets/")
        print("\nEach sub-dataset contains samples with specific prompt patterns:")
        print("  - neutral: Just the question")
        print("  - positive_correct_answer: User suggests correct answer")
        print("  - positive_incorrect_answer: User suggests wrong answer")
        print("  - negative_correct_answer: User doubts correct answer")
        print("  - etc.")


if __name__ == "__main__":
    main()