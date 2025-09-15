#!/usr/bin/env python3
"""
Script to re-upload the dataset with proper configurations.
"""

import os
import json
from pathlib import Path
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import HfApi, login, create_repo
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def reorganize_and_upload(source_repo: str = "Noddybear/lies",
                          target_repo: str = "Noddybear/lies-v2",
                          token: str = None):
    """
    Reorganize the dataset into proper configurations and upload.

    Args:
        source_repo: Current dataset repo (with single config)
        target_repo: Where to upload the reorganized dataset
        token: HF API token
    """

    # Login
    if token or os.environ.get("HF_TOKEN"):
        login(token=token or os.environ.get("HF_TOKEN"))

    api = HfApi()

    # Load the current dataset
    logger.info(f"Loading dataset from {source_repo}...")
    dataset = load_dataset(source_repo)

    # Get all unique folds
    folds = set()
    for split in dataset.keys():
        for sample in dataset[split]:
            if sample['fold'] and sample['fold'] != 'single':
                folds.add(sample['fold'])

    logger.info(f"Found {len(folds)} folds: {sorted(folds)}")

    # Create target repo if it doesn't exist
    try:
        api.create_repo(repo_id=target_repo, repo_type="dataset", exist_ok=True)
        logger.info(f"Created/verified repository: {target_repo}")
    except Exception as e:
        logger.error(f"Error creating repo: {e}")

    # Process each fold as a separate configuration
    for fold_name in sorted(folds):
        logger.info(f"\nProcessing fold: {fold_name}")

        # Get model and aggregation info from first sample
        first_sample = None
        for sample in dataset['train']:
            if sample['fold'] == fold_name:
                first_sample = sample
                break

        if not first_sample:
            logger.warning(f"No samples found for fold {fold_name}")
            continue

        model_name = first_sample['model'].replace("/", "-")
        aggregation = first_sample['aggregation']

        # Create configuration name
        config_name = f"{model_name}_{aggregation}_{fold_name}"
        logger.info(f"Configuration name: {config_name}")

        # Filter dataset for this fold
        fold_splits = {}

        for split_name in dataset.keys():
            # Filter samples for this fold
            fold_data = dataset[split_name].filter(lambda x: x['fold'] == fold_name)

            if len(fold_data) > 0:
                fold_splits[split_name] = fold_data
                logger.info(f"  {split_name}: {len(fold_data)} samples")

        # Create DatasetDict for this configuration
        if fold_splits:
            fold_dataset = DatasetDict(fold_splits)

            # Upload this configuration
            try:
                logger.info(f"Uploading configuration: {config_name}")
                fold_dataset.push_to_hub(
                    repo_id=target_repo,
                    config_name=config_name,
                    commit_message=f"Add {config_name} configuration"
                )
                logger.info(f"✓ Successfully uploaded {config_name}")
            except Exception as e:
                logger.error(f"✗ Failed to upload {config_name}: {e}")

    # Create a comprehensive README
    logger.info("\nCreating dataset card...")
    create_dataset_card(target_repo, folds, api)

    logger.info(f"\n✅ Dataset reorganization complete!")
    logger.info(f"View at: https://huggingface.co/datasets/{target_repo}")


def create_dataset_card(repo_id: str, folds: set, api: HfApi):
    """Create a comprehensive dataset card."""

    # Build configuration table
    config_table = "| Configuration | Model | Aggregation | Fold | Description |\n"
    config_table += "|--------------|-------|-------------|------|-------------|\n"

    for fold in sorted(folds):
        # Assuming openai/gpt-4o and generalization_map_3 for now
        # You can extract this from the actual data
        config_name = f"openai-gpt-4o_generalization_map_3_{fold}"
        config_table += f"| `{config_name}` | openai/gpt-4o | generalization_map_3 | {fold} | Cross-validation fold for '{fold}' category |\n"

    card_content = f"""---
language:
- en
license: mit
task_categories:
- text-classification
pretty_name: Lie Detection Dataset
tags:
- lie-detection
- truthfulness
- fine-tuning
- safety
configs:
{chr(10).join([f'- config_name: openai-gpt-4o_generalization_map_3_{fold}' for fold in sorted(folds)])}
---

# Lie Detection Dataset

This dataset contains lie detection samples for fine-tuning language models to detect when they are lying.

## Available Configurations

Each configuration represents a cross-validation fold for testing generalization:

{config_table}

## Usage

```python
from datasets import load_dataset

# Load a specific fold
dataset = load_dataset("{repo_id}", "openai-gpt-4o_generalization_map_3_sycophancy")

# Access the splits
train_data = dataset["train"]
val_data = dataset["validation"]
test_data = dataset["test"]
```

## Dataset Structure

Each sample contains:
- `sample_id`: Unique identifier
- `task`: Task name (e.g., "sycophancy")
- `category`: Task category from aggregation
- `did_lie`: Whether the model lied (boolean)
- `lie_detection_prompt`: The prompt asking about lying
- `trace`: Full conversation history
- `model`: Model that generated responses
- And more...

## Cross-validation Methodology

Each configuration holds out one task category for testing:
- **Train**: Samples from all other categories
- **Validation**: Subset from same categories as train
- **Test**: Only samples from the held-out category

This tests whether models can generalize lie detection to new task types.
"""

    # Upload the card
    api.upload_file(
        path_or_fileobj=card_content.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add comprehensive dataset card"
    )


def create_yaml_config(repo_id: str, folds: set, api: HfApi):
    """Create a YAML configuration file for the dataset."""

    yaml_lines = []

    for fold in sorted(folds):
        config_name = f"openai-gpt-4o_generalization_map_3_{fold}"
        yaml_lines.extend([
            f"- config_name: {config_name}",
            "  data_files:",
            f"    - split: train",
            f"      pattern: data/{config_name}/train-*.parquet",
            f"    - split: validation",
            f"      pattern: data/{config_name}/validation-*.parquet",
            f"    - split: test",
            f"      pattern: data/{config_name}/test-*.parquet",
            ""
        ])

    yaml_content = "configs:\n" + "\n".join(yaml_lines)

    # This would need to be added to the README YAML header
    logger.info("YAML configuration created (add to README header)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reorganize dataset with proper configs")
    parser.add_argument("--source-repo", default="Noddybear/lies", help="Source repository")
    parser.add_argument("--target-repo", default="Noddybear/lies", help="Target repository")
    parser.add_argument("--token", help="HF API token")
    parser.add_argument("--same-repo", action="store_true", help="Update the same repository")

    args = parser.parse_args()

    if args.same_repo:
        # Use the same repo for both source and target
        # Note: This might require deleting existing data first
        reorganize_and_upload(args.source_repo, args.source_repo, args.token)
    else:
        reorganize_and_upload(args.source_repo, args.target_repo, args.token)