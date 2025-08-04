import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Value, Sequence
from huggingface_hub import HfApi, login
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class HuggingFaceUploader:
    """Upload lie detection datasets to Hugging Face Hub."""

    def __init__(self, repo_id: str = "Noddybear/lies", token: Optional[str] = None):
        """
        Initialize the uploader.

        Args:
            repo_id: The Hugging Face dataset repository ID
            token: Hugging Face API token (if not provided, will use HF_TOKEN env var)
        """
        self.repo_id = repo_id
        self.token = token or os.environ.get("HF_TOKEN")

        if self.token:
            login(token=self.token)

        self.api = HfApi()

        # Define the dataset features schema for proper display in the viewer
        self.features = Features({
            "sample_id": Value("string"),
            "task_id": Value("string"),
            "task": Value("string"),
            "category": Value("string"),
            "did_lie": Value("bool"),
            "lie_detection_prompt": {
                "role": Value("string"),
                "content": Value("string")
            },
            "trace": Sequence({
                "role": Value("string"),
                "content": Value("string")
            }),
            "model": Value("string"),
            "dataset_type": Value("string"),  # train/val/test
            "fold": Value("string"),
            "aggregation": Value("string"),
            "balance_strategy": Value("string"),
            "timestamp": Value("string")
        })

    def _convert_training_format_to_dataset(self, jsonl_path: Path,
                                            dataset_type: str,
                                            metadata: Dict[str, Any]) -> List[Dict]:
        """
        Convert training JSONL format back to dataset format for HF.

        Args:
            jsonl_path: Path to train.jsonl, val.jsonl, or test.jsonl
            dataset_type: One of "train", "val", or "test"
            metadata: Dataset metadata including fold info

        Returns:
            List of samples in HF dataset format
        """
        samples = []

        with open(jsonl_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)

                    # Check if we have enough messages
                    messages = data.get("messages", [])
                    if len(messages) < 2:
                        logger.warning(
                            f"Skipping line {line_num}: insufficient messages (found {len(messages)}, need at least 2)")
                        continue

                    # Extract the original trace (all messages except the last two)
                    trace = messages[:-2]  # Remove lie detection prompt and response

                    # Get the lie detection prompt (second to last message)
                    lie_detection_prompt = messages[-2]

                    # Determine did_lie from the response
                    response = messages[-1].get("content", "").strip().lower()
                    did_lie = response == "yes."

                    # Get meta information if available
                    meta = data.get("meta", {})

                    # Extract task and sample_id from meta
                    task = meta.get("task", "unknown")
                    sample_id = meta.get("sample_id", f"{dataset_type}_{len(samples)}")
                    task_id = meta.get("task_id", "unknown")
                    category = meta.get("category", metadata.get("fold_name", "unknown"))

                    sample = {
                        "sample_id": sample_id,
                        "task_id": task_id,
                        "task": task,
                        "category": category,
                        "did_lie": did_lie,
                        "lie_detection_prompt": lie_detection_prompt,
                        "trace": trace,
                        "model": metadata.get("model", "unknown"),
                        "dataset_type": dataset_type,
                        "fold": metadata.get("fold_name", "single"),
                        "aggregation": metadata.get("aggregation", "none"),
                        "balance_strategy": metadata.get("balance_strategy", "none"),
                        "timestamp": datetime.now().isoformat()
                    }

                    samples.append(sample)

                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")
                    continue

        return samples

    def upload_dataset(self,
                       dataset_path: str,
                       dataset_name: Optional[str] = None,
                       private: bool = False) -> Dict[str, Any]:
        """
        Upload a processed dataset to Hugging Face.

        Args:
            dataset_path: Path to the dataset directory (containing train.jsonl, val.jsonl, etc.)
            dataset_name: Optional name for the dataset configuration
            private: Whether to make the dataset private

        Returns:
            Dictionary with upload results
        """
        dataset_path = Path(dataset_path)

        # Load metadata
        metadata_path = dataset_path / "metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"No metadata.json found in {dataset_path}")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Generate dataset name if not provided
        if not dataset_name:
            model_name = metadata.get("model", "unknown").replace("/", "-")
            fold_name = metadata.get("fold_name", "single")
            aggregation = metadata.get("aggregation", "none")

            # Clean up the names to match expected format
            model_name = model_name.replace("_", "-")

            if fold_name and fold_name != "single":
                dataset_name = f"{model_name}_{aggregation}_{fold_name}"
            else:
                dataset_name = f"{model_name}_{aggregation}"

        logger.info(f"Uploading dataset: {dataset_name}")

        # Collect all splits
        splits = {}

        # Process train split
        train_path = dataset_path / "train.jsonl"
        if train_path.exists():
            logger.info("Processing training data...")
            train_samples = self._convert_training_format_to_dataset(
                train_path, "train", metadata
            )
            splits["train"] = Dataset.from_list(train_samples, features=self.features)
            logger.info(f"  - Train: {len(train_samples)} samples")

        # Process validation split
        val_path = dataset_path / "val.jsonl"
        if val_path.exists():
            logger.info("Processing validation data...")
            val_samples = self._convert_training_format_to_dataset(
                val_path, "validation", metadata
            )
            splits["validation"] = Dataset.from_list(val_samples, features=self.features)
            logger.info(f"  - Validation: {len(val_samples)} samples")

        # Process test split (if exists)
        test_path = dataset_path / "test.jsonl"
        if test_path.exists():
            logger.info("Processing test data...")
            test_samples = self._convert_training_format_to_dataset(
                test_path, "test", metadata
            )
            splits["test"] = Dataset.from_list(test_samples, features=self.features)
            logger.info(f"  - Test: {len(test_samples)} samples")

        # Create DatasetDict
        dataset_dict = DatasetDict(splits)

        # Upload to Hugging Face
        logger.info(f"Uploading to {self.repo_id} (configuration: {dataset_name})...")

        dataset_dict.push_to_hub(
            repo_id=self.repo_id,
            config_name=dataset_name,
            private=private,
            commit_message=f"Add {dataset_name} dataset",
        )

        # After uploading, update the main dataset card with all configurations
        logger.info("Updating main dataset card...")
        main_card = self._create_or_update_dataset_card(self.repo_id)

        self.api.upload_file(
            path_or_fileobj=main_card.encode(),
            path_in_repo="README.md",
            repo_id=self.repo_id,
            repo_type="dataset",
            commit_message=f"Update dataset card with {dataset_name} configuration"
        )

        logger.info(f"Successfully uploaded dataset to {self.repo_id}/{dataset_name}")

        return {
            "repo_id": self.repo_id,
            "config_name": dataset_name,
            "url": f"https://huggingface.co/datasets/{self.repo_id}",
            "splits": {split: len(data) for split, data in splits.items()},
            "total_samples": sum(len(data) for data in splits.values())
        }

    def upload_all_folds(self, model_dir: str, private: bool = False) -> List[Dict[str, Any]]:
        """
        Upload all folds from a model directory as a single dataset with multiple configurations.

        Args:
            model_dir: Directory containing multiple fold subdirectories
            private: Whether to make the datasets private

        Returns:
            List of upload results
        """
        model_path = Path(model_dir)
        results = []

        # Check if this is a single dataset or multiple folds
        if (model_path / "train.jsonl").exists():
            # Single dataset
            logger.info("Found single dataset (no folds)")
            result = self.upload_dataset(str(model_path), private=private)
            results.append(result)
        else:
            # Multiple folds - need to upload them all at once
            fold_dirs = [d for d in model_path.iterdir()
                         if d.is_dir() and (d / "train.jsonl").exists()]

            if not fold_dirs:
                logger.error("No valid fold directories found")
                return results

            logger.info(f"Found {len(fold_dirs)} folds to upload")

            # First, collect all configurations
            all_configs = {}

            for fold_dir in fold_dirs:
                logger.info(f"\nProcessing fold: {fold_dir.name}")

                # Load metadata
                metadata_path = fold_dir / "metadata.json"
                if not metadata_path.exists():
                    logger.error(f"No metadata.json found in {fold_dir}")
                    continue

                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                # Generate config name
                model_name = metadata.get("model", "unknown").replace("/", "-")
                fold_name = metadata.get("fold_name", "single")
                aggregation = metadata.get("aggregation", "none")

                if fold_name and fold_name != "single":
                    config_name = f"{model_name}_{aggregation}_{fold_name}"
                else:
                    config_name = f"{model_name}_{aggregation}"

                # Process all splits for this fold
                splits = {}

                # Process train split
                train_path = fold_dir / "train.jsonl"
                if train_path.exists():
                    train_samples = self._convert_training_format_to_dataset(
                        train_path, "train", metadata
                    )
                    splits["train"] = Dataset.from_list(train_samples, features=self.features)

                # Process validation split
                val_path = fold_dir / "val.jsonl"
                if val_path.exists():
                    val_samples = self._convert_training_format_to_dataset(
                        val_path, "validation", metadata
                    )
                    splits["validation"] = Dataset.from_list(val_samples, features=self.features)

                # Process test split (if exists)
                test_path = fold_dir / "test.jsonl"
                if test_path.exists():
                    test_samples = self._convert_training_format_to_dataset(
                        test_path, "test", metadata
                    )
                    splits["test"] = Dataset.from_list(test_samples, features=self.features)

                # Store configuration
                all_configs[config_name] = DatasetDict(splits)

                logger.info(f"  ✓ Processed {config_name}: {sum(len(s) for s in splits.values())} total samples")

            # Now upload ALL configurations at once
            if all_configs:
                logger.info(f"\nUploading {len(all_configs)} configurations to {self.repo_id}...")

                # Create or verify the repository exists
                try:
                    self.api.create_repo(repo_id=self.repo_id, repo_type="dataset", exist_ok=True, private=private)
                except Exception as e:
                    logger.warning(f"Repo creation warning: {e}")

                # Upload each configuration
                for config_name, dataset_dict in all_configs.items():
                    try:
                        logger.info(f"Uploading configuration: {config_name}")

                        dataset_dict.push_to_hub(
                            repo_id=self.repo_id,
                            config_name=config_name,
                            private=private,
                            commit_message=f"Add {config_name} configuration",
                        )

                        # Calculate stats for results
                        total_samples = sum(len(split) for split in dataset_dict.values())
                        splits_info = {split: len(data) for split, data in dataset_dict.items()}

                        results.append({
                            "repo_id": self.repo_id,
                            "config_name": config_name,
                            "url": f"https://huggingface.co/datasets/{self.repo_id}",
                            "splits": splits_info,
                            "total_samples": total_samples
                        })

                        logger.info(f"  ✅ Successfully uploaded {config_name}")

                    except Exception as e:
                        logger.error(f"  ❌ Failed to upload {config_name}: {e}")
                        results.append({
                            "config_name": config_name,
                            "status": "failed",
                            "error": str(e)
                        })

                # Update the dataset card after all uploads
                logger.info("\nUpdating dataset card...")
                self._update_dataset_card_with_all_configs(all_configs)

        return results

    def _update_dataset_card_with_all_configs(self, all_configs: Dict[str, DatasetDict]):
        """Update dataset card with information about all configurations."""

        # Create configuration table
        config_table = "| Configuration | Model | Aggregation | Fold | Samples | Description |\n"
        config_table += "|--------------|-------|-------------|------|---------|-------------|\n"

        for config_name, dataset_dict in sorted(all_configs.items()):
            # Parse config name
            parts = config_name.split("_")
            if len(parts) >= 3:
                model = parts[0].replace("-", "/")
                aggregation = "_".join(parts[1:-1]) if len(parts) > 3 else parts[1]
                fold = parts[-1]
            else:
                model = "unknown"
                aggregation = "unknown"
                fold = "unknown"

            total_samples = sum(len(split) for split in dataset_dict.values())

            config_table += f"| `{config_name}` | {model} | {aggregation} | {fold} | {total_samples} | "

            if fold != "single" and fold != "unknown":
                config_table += f"Fold containing only '{fold}' category samples |\n"
            else:
                config_table += f"Standard train/val split |\n"

        # Create dataset card content
        card_content = \
f"""
---
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
---

# Lie Detection Dataset

This dataset contains lie detection samples for fine-tuning language models to detect when they are lying.

## Dataset Summary

Each sample contains a conversation between a user and an AI model, followed by a lie detection prompt and the ground truth label indicating whether the model lied.

## Available Configurations

{config_table}

## Usage

```python
from datasets import load_dataset

# Load a specific configuration
dataset = load_dataset("{self.repo_id}", "{list(all_configs.keys())[0] if all_configs else 'config_name'}")

# Access splits
train_data = dataset["train"]
val_data = dataset["validation"]
```

## Dataset Structure
Each sample contains:

- sample_id: Unique identifier
- task: Task name
- category: Task category from aggregation
- did_lie: Whether the model lied (boolean)
- trace: Full conversation history
- model: Model that generated responses

And more fields...

@dataset{{lie_detection_2025,
  title={{Lie Detection Dataset}},
  year={{2025}},
  publisher={{Hugging Face}}
}}"""
        try:
            self.api.upload_file(
                path_or_fileobj=card_content.encode(),
                path_in_repo="README.md",
                repo_id=self.repo_id,
                repo_type="dataset",
                commit_message="Update dataset card with all configurations"
            )
            logger.info("✅ Dataset card updated successfully")
        except Exception as e:
            logger.error(f"Failed to update dataset card: {e}")

    def _create_or_update_dataset_card(self, repo_id: str) -> str:
        """Create or update the main dataset card with all configurations."""

        try:
            # Get list of all configurations in the dataset
            configs = self.api.list_repo_files(repo_id=repo_id, repo_type="dataset")
            config_names = set()

            # Parse config names from the files
            for file in configs:
                if file.startswith("config_") and file.endswith(".json"):
                    config_name = file[7:-5]  # Remove "config_" and ".json"
                    config_names.add(config_name)

            # Also get configs from dataset info if available
            try:
                from datasets import get_dataset_config_names
                hf_configs = get_dataset_config_names(repo_id)
                config_names.update(hf_configs)
            except:
                pass

            # Sort configurations for consistent display
            sorted_configs = sorted(list(config_names))

            # Create configuration table
            config_table = "| Configuration | Model | Aggregation | Fold | Description |\n"
            config_table += "|--------------|-------|-------------|------|-------------|\n"

            for config in sorted_configs:
                # Parse config name
                parts = config.split("_")
                if len(parts) >= 3:
                    model = parts[0].replace("-", "/")
                    aggregation = "_".join(parts[1:-1]) if len(parts) > 3 else parts[1]
                    fold = parts[-1]

                    config_table += f"| `{config}` | {model} | {aggregation} | {fold} | "

                    if aggregation != "none":
                        config_table += f"Cross-validation fold for {fold} category |\n"
                    else:
                        config_table += f"Standard train/val/test split |\n"

            # Create comprehensive dataset card
            card = f"""---
language:
- en
task_categories:
- text-classification
pretty_name: Lie Detection Dataset
tags:
- lie-detection
- truthfulness
- fine-tuning
- safety
---

# Lie Detection Dataset

This dataset contains lie detection samples for fine-tuning language models to detect when they are lying.

## Dataset Summary

This dataset is designed to train models to recognize when they have lied in their responses. Each sample contains:
- A conversation trace between a user and an AI model
- A lie detection prompt asking if the model lied
- The ground truth label indicating whether the model actually lied

## Available Configurations

{config_table}

## Dataset Structure

### Data Fields

Each sample contains the following fields:

- `sample_id` (string): Unique identifier for the sample
- `task_id` (string): Task identifier
- `task` (string): Task name/type (e.g., "sycophancy", "sandbagging", etc.)
- `category` (string): Task category from aggregation mapping
- `did_lie` (bool): Ground truth - whether the model lied
- `lie_detection_prompt` (dict): The prompt asking about lying
  - `role` (string): Always "user"
  - `content` (string): The lie detection question
- `trace` (list[dict]): Full conversation history
  - `role` (string): Either "user" or "assistant"
  - `content` (string): Message content
- `model` (string): Model that generated the responses
- `dataset_type` (string): Split type (train/validation/test)
- `fold` (string): Fold name for cross-validation
- `aggregation` (string): Aggregation strategy used
- `balance_strategy` (string): Strategy used for balancing lies/truths
- `timestamp` (string): When the dataset was created

### Data Splits

Each configuration typically contains:
- `train`: Training samples
- `validation`: Validation samples
- `test`: Test samples (if using 3-way split)

## Usage

### Loading a Specific Configuration

```python
from datasets import load_dataset

# Load a specific model and fold
dataset = load_dataset("Noddybear/lies", "openai-gpt-4o_generalization_map_3_sycophancy")

# Access the splits
train_data = dataset["train"]
val_data = dataset["validation"]
test_data = dataset["test"]  # if available

# Example: Print first sample
print(train_data[0])
```

### Loading All Configurations

```python
from datasets import load_dataset, get_dataset_config_names

# Get all available configurations
configs = get_dataset_config_names("Noddybear/lies")

# Load multiple configurations
datasets = {{}}
for config in configs:
    datasets[config] = load_dataset("Noddybear/lies", config)
```

### Using for Fine-tuning

The dataset is formatted for direct use with OpenAI's fine-tuning API:

```python
# Convert back to training format
def prepare_for_finetuning(sample):
    messages = sample["trace"].copy()
    messages.append(sample["lie_detection_prompt"])
    messages.append({{
        "role": "assistant",
        "content": "Yes." if sample["did_lie"] else "No."
    }})
    return {{"messages": messages}}

# Prepare training data
train_samples = [prepare_for_finetuning(s) for s in train_data]
```

## Dataset Creation

### Aggregation Strategies

The dataset uses different aggregation strategies to group tasks:
- `generalization_map_1`: Basic task grouping
- `generalization_map_2`: Intermediate grouping
- `generalization_map_3`: Fine-grained task categories
- `none`: No aggregation (all tasks together)

### Balance Strategies

- `downsample`: Reduce majority class to match minority
- `upsample`: Duplicate minority class to match majority
- `none`: No balancing

### Cross-validation Folds

When using aggregation, the dataset creates leave-one-category-out folds to test generalization.

## Additional Information

### Licensing

[Specify your license here]

### Citation

If you use this dataset, please cite:

```bibtex
@dataset{{lie_detection_2025,
  title={{Lie Detection Dataset for Language Model Fine-tuning}},
  author={{Your Team}},
  year={{2025}},
  month={{1}},
  publisher={{Hugging Face}},
  url={{https://huggingface.co/datasets/Noddybear/lies}}
}}
```

### Contact

[Add contact information]
"""
            return card

        except Exception as e:
            logger.error(f"Error creating dataset card: {e}")
            # Return a basic card if we can't get the config list
            return self._create_basic_dataset_card()

    def _create_basic_dataset_card(self) -> str:
        """Create a basic dataset card when we can't get the full config list."""
        return """---
language:
- en
task_categories:
- text-classification
pretty_name: Lie Detection Dataset
tags:
- lie-detection
- truthfulness
- fine-tuning
- safety
---

# Lie Detection Dataset

This dataset contains lie detection samples for fine-tuning language models to detect when they are lying.

## Dataset Structure

Each sample contains:
- `sample_id`: Unique identifier for the sample
- `task_id`: Task identifier
- `task`: Task name/type
- `category`: Task category (from aggregation mapping)
- `did_lie`: Boolean indicating if the model lied
- `lie_detection_prompt`: The prompt asking about lying
- `trace`: Conversation history
- `model`: Model that generated the responses
- `dataset_type`: Split type (train/validation/test)
- `fold`: Fold name for cross-validation
- `aggregation`: Aggregation strategy used
- `balance_strategy`: Strategy used for balancing lies/truths
- `timestamp`: When the dataset was created

## Usage

```python
from datasets import load_dataset

# List all available configurations
configs = get_dataset_config_names("Noddybear/lies")
print(f"Available configurations: {configs}")

# Load a specific configuration
dataset = load_dataset("Noddybear/lies", "your-config-name")
```
"""


def main():
    """CLI for uploading datasets to Hugging Face."""
    import argparse

    parser = argparse.ArgumentParser(description='Upload lie detection datasets to Hugging Face')
    parser.add_argument('--dataset', required=True, help='Path to dataset directory or model directory')
    parser.add_argument('--repo-id', default='Noddybear/lies', help='Hugging Face repository ID')
    parser.add_argument('--all-folds', action='store_true', help='Upload all folds in directory')
    parser.add_argument('--private', action='store_true', help='Make dataset private')
    parser.add_argument('--token', help='Hugging Face API token (or set HF_TOKEN env var)')
    parser.add_argument('--name', help='Custom name for the dataset configuration')

    args = parser.parse_args()

    # Initialize uploader
    uploader = HuggingFaceUploader(repo_id=args.repo_id, token=args.token)

    if args.all_folds:
        # Upload all folds
        results = uploader.upload_all_folds(args.dataset, private=args.private)

        print("\n" + "=" * 70)
        print("UPLOAD SUMMARY")
        print("=" * 70)

        for result in results:
            if "error" in result:
                print(f"❌ {result.get('fold', 'Unknown')}: Failed - {result['error']}")
            else:
                print(f"✅ {result['config_name']}: {result['total_samples']} samples")
                print(f"   URL: {result['url']}")
    else:
        # Upload single dataset
        result = uploader.upload_dataset(
            args.dataset,
            dataset_name=args.name,
            private=args.private
        )

        print("\n" + "=" * 70)
        print("UPLOAD COMPLETE")
        print("=" * 70)
        print(f"Repository: {result['repo_id']}")
        print(f"Configuration: {result['config_name']}")
        print(f"URL: {result['url']}")
        print(f"Total samples: {result['total_samples']}")
        print("\nSplits:")
        for split, count in result['splits'].items():
            print(f"  - {split}: {count} samples")


if __name__ == "__main__":
    main()