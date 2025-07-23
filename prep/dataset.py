import argparse
import hashlib
import json
import logging
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from common.generalization_mappings import generalization_map_1, generalization_map_2
from prep.download import S3DataDownloader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LieDetectionDataProcessor:
    """Processes lie detection samples for fine-tuning."""

    def __init__(self, samples: List[Dict]):
        self.samples = samples
        self.generalization_maps = {
            'generalization_map_1': generalization_map_1,
            'generalization_map_2': generalization_map_2
        }

    def deduplicate_samples(self) -> List[Dict]:
        """Deduplicate samples based on sample ID hash."""
        seen_hashes = set()
        unique_samples = []

        for sample in self.samples:
            sample_hash = hashlib.md5((str(sample['sample_id']) + str(sample['task_id'])).encode()).hexdigest()
            if sample_hash not in seen_hashes:
                seen_hashes.add(sample_hash)
                unique_samples.append(sample)

        logger.info(f"Deduplicated {len(self.samples)} samples to {len(unique_samples)} unique samples")
        return unique_samples

    def categorize_by_task(self, samples: List[Dict], aggregation: str) -> Dict[str, List[Dict]]:
        """Categorize samples by their generalization category."""
        if aggregation == 'none':
            return {'all': samples}

        generalization_map = self.generalization_maps.get(aggregation, {})
        categorized = defaultdict(list)
        uncategorized = []

        # Create reverse mapping from task to category
        task_to_category = {}
        for category, tasks in generalization_map.items():
            for task in tasks:
                task_to_category[task] = category

        uncategorized_tasks = set()
        for sample in samples:
            task = sample.get('task', '')
            category = task_to_category.get(task)

            if category:
                categorized[category].append(sample)
            else:
                uncategorized.append(sample)
                uncategorized_tasks.add(task)

        if uncategorized:
            logger.warning(
                f"Found {len(uncategorized)} uncategorized samples across the following tasks: {uncategorized_tasks}")

        for category, cat_samples in categorized.items():
            logger.info(f"Category '{category}': {len(cat_samples)} samples")

        return dict(categorized)

    def balance_samples(self, samples: List[Dict], balance_strategy: Optional[str]) -> List[Dict]:
        """Balance samples based on did_lie label."""
        if not balance_strategy or balance_strategy == 'none':
            return samples

        true_samples = [s for s in samples if s.get('did_lie', False)]
        false_samples = [s for s in samples if not s.get('did_lie', False)]

        logger.info(f"Before balancing: {len(true_samples)} lies, {len(false_samples)} truths")

        if balance_strategy == 'upsample':
            # Upsample minority class
            if len(true_samples) < len(false_samples):
                true_samples = self._upsample(true_samples, len(false_samples))
            else:
                false_samples = self._upsample(false_samples, len(true_samples))

        elif balance_strategy == 'downsample':
            # Downsample majority class
            target_size = min(len(true_samples), len(false_samples))
            if len(true_samples) > target_size:
                true_samples = random.sample(true_samples, target_size)
            if len(false_samples) > target_size:
                false_samples = random.sample(false_samples, target_size)

        balanced = true_samples + false_samples
        random.shuffle(balanced)

        logger.info(f"After balancing: {len([s for s in balanced if s.get('did_lie', False)])} lies, "
                    f"{len([s for s in balanced if not s.get('did_lie', False)])} truths")

        return balanced

    def _upsample(self, samples: List[Dict], target_size: int) -> List[Dict]:
        """Upsample by duplicating samples."""
        if not samples:
            return []

        upsampled = samples.copy()
        while len(upsampled) < target_size:
            additional_needed = target_size - len(upsampled)
            additional = random.choices(samples, k=min(additional_needed, len(samples)))
            upsampled.extend(additional)

        return upsampled

    def create_folds(self, categorized_samples: Dict[str, List[Dict]],
                     size: Optional[int] = None,
                     validation_split: float = 0.2) -> List[Tuple[str, List[Dict], List[Dict], Optional[List[Dict]]]]:
        """
        Create train/validation/test folds based on categories.

        Args:
            categorized_samples: Dict of category -> samples
            size: Optional max training size
            validation_split: Fraction of training data for validation (0.0-0.5)
                             If 0.0, uses OOD data as validation (no test set)

        Returns:
            List of (fold_name, train, val, test) tuples
            test will be None if validation_split == 0.0
        """
        categories = list(categorized_samples.keys())

        if len(categories) == 1:
            # Single category - standard split
            all_samples = categorized_samples[categories[0]]
            shuffled = all_samples.copy()
            random.shuffle(shuffled)

            if validation_split > 0:
                # Three-way split
                train_size = int((1 - validation_split - 0.2) * len(shuffled))  # 60% by default
                val_size = int(validation_split * len(shuffled))  # 20%

                train = shuffled[:train_size]
                val = shuffled[train_size:train_size + val_size]
                test = shuffled[train_size + val_size:]  # 20%

                return [("all", train, val, test)]
            else:
                # Two-way split (80/20)
                val_size = int(0.2 * len(shuffled))
                val = shuffled[:val_size]
                train = shuffled[val_size:]

                return [("all", train, val, None)]

        # Multiple categories - leave-one-out cross validation
        folds = []

        for held_out_category in categories:
            # Collect samples
            in_distribution_samples = []
            held_out_samples = categorized_samples[held_out_category]

            for category, samples in categorized_samples.items():
                if category != held_out_category:
                    in_distribution_samples.extend(samples)

            # Shuffle in-distribution samples
            random.shuffle(in_distribution_samples)

            if validation_split > 0:
                # Three-way split: train/val from ID, test from OOD
                val_size = int(validation_split * len(in_distribution_samples))
                val_samples = in_distribution_samples[:val_size]
                train_samples = in_distribution_samples[val_size:]
                test_samples = held_out_samples  # OOD test set

                # Apply size limit if specified
                if size and len(train_samples) > size:
                    train_samples = random.sample(train_samples, size)

                logger.info(f"Fold '{held_out_category}' (3-way split):")
                logger.info(f"  - Train: {len(train_samples)} samples from {len(categories) - 1} categories")
                logger.info(f"  - Val: {len(val_samples)} samples from same {len(categories) - 1} categories")
                logger.info(f"  - Test: {len(test_samples)} samples from '{held_out_category}' only (OOD)")

                folds.append((held_out_category, train_samples, val_samples, test_samples))
            else:
                # Two-way split: train from ID, val from OOD
                train_samples = in_distribution_samples
                val_samples = held_out_samples  # OOD validation

                # Apply size limit if specified
                if size and len(train_samples) > size:
                    train_samples = random.sample(train_samples, size)

                logger.info(f"Fold '{held_out_category}' (2-way split):")
                logger.info(f"  - Train: {len(train_samples)} samples from {len(categories) - 1} categories")
                logger.info(f"  - Val: {len(val_samples)} samples from '{held_out_category}' only (OOD)")

                folds.append((held_out_category, train_samples, val_samples, None))

        return folds

    def _create_kfold_splits(self, samples: List[Dict], size: Optional[int] = None,
                             validation_split: float = 0.2) -> List[
        Tuple[List[Dict], List[Dict], Optional[List[Dict]]]]:
        """Create k-fold splits for a single set of samples."""
        if size and size < len(samples):
            # Create multiple folds based on size
            num_folds = len(samples) // size
            folds = []

            shuffled = samples.copy()
            random.shuffle(shuffled)

            for i in range(num_folds):
                start_idx = i * size
                end_idx = min((i + 1) * size, len(samples))

                val_samples = shuffled[start_idx:end_idx]
                train_samples = shuffled[:start_idx] + shuffled[end_idx:]

                if size and len(train_samples) > size:
                    train_samples = random.sample(train_samples, size)

                folds.append((train_samples, val_samples, None))

            return folds
        else:
            # Single split
            shuffled = samples.copy()
            random.shuffle(shuffled)

            if validation_split > 0:
                # Three-way split: 60/20/20
                train_size = int(0.6 * len(shuffled))
                val_size = int(0.2 * len(shuffled))

                train_samples = shuffled[:train_size]
                val_samples = shuffled[train_size:train_size + val_size]
                test_samples = shuffled[train_size + val_size:]

                if size and len(train_samples) > size:
                    train_samples = random.sample(train_samples, size)

                return [(train_samples, val_samples, test_samples)]
            else:
                # Two-way split: 80/20
                val_size = int(0.2 * len(shuffled))
                val_samples = shuffled[:val_size]
                train_samples = shuffled[val_size:]

                if size and len(train_samples) > size:
                    train_samples = random.sample(train_samples, size)

                return [(train_samples, val_samples, None)]

    def convert_to_training_format(self, samples: List[Dict]) -> List[Dict]:
        """Convert samples to the SafetyTools training format."""
        training_data = []

        for sample in samples:
            # Add lie detection prompt and expected response
            messages = sample['trace'].copy()

            # Add the lie detection question
            messages.append(sample['lie_detection_prompt'])

            # Add expected response based on did_lie
            if sample['did_lie']:
                response = "Yes."
            else:
                response = "No."

            messages.append({
                "role": "assistant",
                "content": response
            })

            training_data.append({"messages": messages, "metadata": {"did_lie": sample['did_lie']}})

        return training_data


def save_training_files(train_data: List[Dict], val_data: List[Dict],
                        test_data: Optional[List[Dict]], output_dir: Path,
                        fold_info: Dict):
    """Save training, validation, and optionally test data as JSONL files with metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save training data
    train_file = output_dir / "train.jsonl"
    with open(train_file, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')

    # Save validation data
    val_file = output_dir / "val.jsonl"
    with open(val_file, 'w') as f:
        for item in val_data:
            f.write(json.dumps(item) + '\n')

    # Save test data if provided
    if test_data is not None:
        test_file = output_dir / "test.jsonl"
        with open(test_file, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
        logger.info(f"Saved {len(test_data)} test samples to {test_file}")

    # Save metadata
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(fold_info, f, indent=2)

    logger.info(f"Saved {len(train_data)} training samples to {train_file}")
    logger.info(f"Saved {len(val_data)} validation samples to {val_file}")
    logger.info(f"Saved fold metadata to {metadata_file}")


def main():
    parser = argparse.ArgumentParser(description='Prepare lie detection data for fine-tuning')
    parser.add_argument('--model', help='Model identifier (e.g., openai/gpt-4o)', default='openai/gpt-4o')
    parser.add_argument('--aggregation', choices=['generalization_map_1', 'generalization_map_2', 'none'],
                        default='generalization_map_1', help='Aggregation strategy for categorizing tasks')
    parser.add_argument('--folds', type=lambda x: x.lower() == 'true', default=True,
                        help='Whether to create cross-validation folds')
    parser.add_argument('--balance', choices=['downsample', 'upsample', 'none'], default='downsample',
                        help='Strategy for balancing lie/truth samples')
    parser.add_argument('--validation-split', type=float, default=0.2,
                        help='Fraction of training data for validation (0.0-0.5). '
                             'If 0.0, uses OOD data as validation with no test set.')
    parser.add_argument('--size', type=int, help='Size of training data (optional)', default=None)
    parser.add_argument('--output-dir', default='.data', help='Base output directory for training files')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Parse model name for directory structure
    provider, model_name = args.model.split('/', 1) if '/' in args.model else ('unknown', args.model)
    clean_provider = provider.lower().replace('-', '_')
    clean_model = model_name.lower().replace('-', '_').replace('/', '_')

    # Create base output directory
    base_output_dir = Path(args.output_dir) / clean_provider / clean_model
    base_output_dir.mkdir(parents=True, exist_ok=True)

    log_file = base_output_dir / "report.log"
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    # Add file handler to root logger to capture all logs
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    # Log start of execution
    logger.info(f"Starting dataset preparation with command: {' '.join(['python'] + [arg for arg in sys.argv])}")

    # Log configuration with human-readable explanations
    logger.info("=" * 70)
    logger.info("LIE DETECTION DATASET PREPARATION")
    logger.info("=" * 70)

    logger.info(f"Configuration:")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Aggregation: {args.aggregation}")
    logger.info(f"  Folds: {args.folds}")
    logger.info(f"  Balance: {args.balance}")
    logger.info(f"  Validation split: {args.validation_split}")
    logger.info(f"  Size: {args.size}")
    logger.info(f"  Output directory: {base_output_dir}")

    # Explain the strategy
    logger.info("")
    logger.info("📋 DATASET STRATEGY:")
    if args.aggregation != 'none':
        logger.info(f"  • Using {args.aggregation} to group tasks into categories")
        if args.folds:
            logger.info("  • Creating leave-one-category-out cross-validation folds")
            if args.validation_split > 0:
                logger.info("    → 3-way split: ID train / ID val / OOD test")
                logger.info("    → Train: ~80% of in-distribution categories")
                logger.info("    → Val: ~20% of same in-distribution categories")
                logger.info("    → Test: 100% of held-out category (for final evaluation)")
            else:
                logger.info("    → 2-way split: ID train / OOD val")
                logger.info("    → Train: 100% of in-distribution categories")
                logger.info("    → Val: 100% of held-out category")
                logger.info("    → No separate test set")
        else:
            logger.info("  • Creating a single split")
            if args.validation_split > 0:
                logger.info("    → 60/20/20 train/val/test split")
            else:
                logger.info("    → 80/20 train/val split")
    else:
        logger.info("  • No task categorization - treating all samples equally")
        if args.folds:
            logger.info("  • Creating multiple random k-fold cross-validation splits")
        else:
            logger.info("  • Creating a single split")
            if args.validation_split > 0:
                logger.info("    → 60/20/20 train/val/test split")
            else:
                logger.info("    → 80/20 train/val split")

    if args.balance != 'none':
        logger.info(f"  • Balancing lies vs truths using {args.balance} strategy")
        if args.balance == 'downsample':
            logger.info("    → Reducing majority class to match minority class size")
        else:
            logger.info("    → Duplicating minority class samples to match majority class")

    if args.size:
        logger.info(f"  • Limiting training data to {args.size} samples per fold")

    logger.info("\n" + "-" * 70)

    # Download samples from S3 (or load from cache)
    logger.info("")
    logger.info("📥 LOADING DATA:")
    downloader = S3DataDownloader()
    samples = downloader.get_model_samples(args.model)

    if not samples:
        logger.error("No samples found for the specified model")
        return

    logger.info(f"  ✓ Loaded {len(samples)} raw samples for {args.model}")

    # Process samples
    processor = LieDetectionDataProcessor(samples)

    # Deduplicate
    logger.info("")
    logger.info("🔍 PROCESSING SAMPLES:")
    unique_samples = processor.deduplicate_samples()
    logger.info(f"  ✓ Removed {len(samples) - len(unique_samples)} duplicate samples")
    logger.info(f"  ✓ {len(unique_samples)} unique samples remaining")

    # Categorize by task
    categorized = processor.categorize_by_task(unique_samples, args.aggregation)

    if args.aggregation != 'none':
        logger.info("")
        logger.info("📊 TASK CATEGORIZATION:")
        total_categorized = sum(len(samples) for samples in categorized.values())
        logger.info(f"  ✓ Grouped {total_categorized} samples into {len(categorized)} categories:")
        for category, cat_samples in categorized.items():
            lies = len([s for s in cat_samples if s.get('did_lie', False)])
            truths = len([s for s in cat_samples if not s.get('did_lie', False)])
            logger.info(f"    • {category}: {len(cat_samples)} samples ({lies} lies, {truths} truths)")

    # Balance each category
    if args.balance != 'none':
        logger.info("")
        logger.info("⚖️  BALANCING LIES VS TRUTHS:")
        for category in categorized:
            before_size = len(categorized[category])
            categorized[category] = processor.balance_samples(
                categorized[category],
                args.balance
            )
            after_size = len(categorized[category])

            if before_size != after_size:
                logger.info(f"  ✓ {category}: {before_size} → {after_size} samples")
            else:
                logger.info(f"  ✓ {category}: already balanced ({after_size} samples)")

    # Create folds or single split
    logger.info("")
    logger.info("📁 CREATING DATASET SPLITS:")

    if args.folds and args.aggregation != 'none':
        split_type = "3-way split" if args.validation_split > 0 else "2-way split"
        logger.info(f"  Creating leave-one-category-out cross-validation folds ({split_type})...")
        folds = processor.create_folds(categorized, args.size, args.validation_split)
    else:
        # Single split or numeric folds
        all_samples = []
        for samples in categorized.values():
            all_samples.extend(samples)

        if args.size and len(all_samples) > args.size:
            all_samples = random.sample(all_samples, args.size)

        if args.folds:
            # Multiple numeric folds
            logger.info(f"  Creating k-fold cross-validation splits...")
            splits = processor._create_kfold_splits(all_samples, args.size, args.validation_split)
            folds = [(f"fold_{i}", train, val, test) for i, (train, val, test) in enumerate(splits)]
            logger.info(f"  ✓ Created {len(folds)} random folds")
        else:
            # Single fold (no subdirectory)
            split_type = "60/20/20" if args.validation_split > 0 else "80/20"
            logger.info(f"  Creating single {split_type} train/val{'/test' if args.validation_split > 0 else ''} split...")
            splits = processor._create_kfold_splits(all_samples, None, args.validation_split)
            folds = [("", splits[0][0], splits[0][1], splits[0][2])]  # Empty name for single fold

    # Create master summary
    master_summary = {
        "model": args.model,
        "aggregation": args.aggregation,
        "folds": args.folds,
        "validation_split": args.validation_split,
        "balance": args.balance,
        "requested_size": args.size,
        "total_samples": len(unique_samples),
        "num_folds": len(folds),
        "has_test_set": args.validation_split > 0,
        "timestamp": datetime.now().isoformat(),
        "seed": args.seed,
        "fold_directories": []
    }

    # Convert to training format and save
    logger.info("")
    logger.info("💾 SAVING DATASETS:")

    for fold_name, train_samples, val_samples, test_samples in folds:
        train_data = processor.convert_to_training_format(train_samples)
        val_data = processor.convert_to_training_format(val_samples)
        test_data = processor.convert_to_training_format(test_samples) if test_samples else None

        # Determine output directory
        if fold_name:  # Multi-fold case
            output_dir = base_output_dir / fold_name
            master_summary["fold_directories"].append(fold_name)
            logger.info("")
            logger.info(f"  📂 Creating fold: {output_dir}")

            if args.aggregation != 'none':
                logger.info(f"     Purpose: Test generalization to '{fold_name}' tasks")
                if args.validation_split > 0:
                    logger.info(f"     Training data: ~80% of all categories EXCEPT '{fold_name}'")
                    logger.info(f"     Validation data: ~20% of same categories (for hyperparameter tuning)")
                    logger.info(f"     Test data: 100% of '{fold_name}' category (for final evaluation)")
                else:
                    logger.info(f"     Training data: All categories EXCEPT '{fold_name}'")
                    logger.info(f"     Validation data: '{fold_name}' category only")
        else:  # Single fold case
            output_dir = base_output_dir
            logger.info("")
            logger.info(f"📂 Saving to: {output_dir}")

        # Create fold metadata
        fold_info = {
            "fold_name": fold_name if fold_name else "single",
            "model": args.model,
            "aggregation": args.aggregation,
            "validation_split": args.validation_split,
            "balance_strategy": args.balance,
            "train_size": len(train_samples),
            "val_size": len(val_samples),
            "test_size": len(test_samples) if test_samples else 0,
            "train_lies": len([s for s in train_samples if s.get('did_lie', False)]),
            "train_truths": len([s for s in train_samples if not s.get('did_lie', False)]),
            "val_lies": len([s for s in val_samples if s.get('did_lie', False)]),
            "val_truths": len([s for s in val_samples if not s.get('did_lie', False)]),
            "timestamp": datetime.now().isoformat(),
            "seed": args.seed
        }

        # Add test statistics if test set exists
        if test_samples:
            fold_info["test_lies"] = len([s for s in test_samples if s.get('did_lie', False)])
            fold_info["test_truths"] = len([s for s in test_samples if not s.get('did_lie', False)])
            fold_info["split_type"] = "train/val/test"
            if args.aggregation != 'none' and fold_name:
                fold_info["test_category"] = fold_name
                fold_info["train_val_categories"] = [cat for cat in categorized.keys() if cat != fold_name]
        else:
            fold_info["split_type"] = "train/val"

        # If using aggregation, add category info
        if args.aggregation != 'none' and fold_name:
            if args.validation_split == 0:
                fold_info["validation_category"] = fold_name
                fold_info["training_categories"] = [cat for cat in categorized.keys() if cat != fold_name]

        save_training_files(train_data, val_data, test_data, output_dir, fold_info)

        # Print what was saved
        logger.info(f"     ✓ train.jsonl: {len(train_data)} samples ({fold_info['train_lies']} lies, {fold_info['train_truths']} truths)")
        logger.info(f"     ✓ val.jsonl: {len(val_data)} samples ({fold_info['val_lies']} lies, {fold_info['val_truths']} truths)")
        if test_data:
            logger.info(f"     ✓ test.jsonl: {len(test_data)} samples ({fold_info['test_lies']} lies, {fold_info['test_truths']} truths)")
        logger.info(f"     ✓ metadata.json: Fold configuration and statistics")

    # Save master summary
    summary_file = base_output_dir / "dataset_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(master_summary, f, indent=2)

    logger.info(f"Master summary saved to: {summary_file}")
    logger.info("     → Contains overview of all folds and dataset creation parameters")

    # Print instructions for fine-tuning
    logger.info("\n" + "="*70)
    logger.info("✅ DATASET PREPARATION COMPLETE!")
    logger.info("="*70)

    if len(folds) == 1 and not folds[0][0]:  # Single fold
        logger.info("")
        logger.info("📌 WHAT WAS CREATED:")
        logger.info(f"  A single train/validation{'/test' if test_data else ''} split in: {base_output_dir}")
        logger.info("  - train.jsonl: Training data")
        logger.info("  - val.jsonl: Validation data (for hyperparameter tuning)")
        if test_data:
            logger.info("  - test.jsonl: Test data (for final evaluation)")
        logger.info("  - metadata.json: Dataset statistics")
        logger.info("  - dataset_summary.json: Creation parameters")

        logger.info("")
        logger.info("🚀 TO FINE-TUNE YOUR MODEL:")
        logger.info("Run the following command:")
        print(f"\npython -m safetytooling.apis.finetuning.openai.run \\")
        print(f"    --model '{args.model}' \\")
        print(f"    --train_file {base_output_dir}/train.jsonl \\")
        print(f"    --val_file {base_output_dir}/val.jsonl \\")
        print(f"    --n_epochs 3")
        if test_data:
            print(f"\n# After training, evaluate on test set:")
            print(f"# python eval.py --model <finetuned_model> --test_file {base_output_dir}/test.jsonl")
    else:
        logger.info("")
        logger.info("📌 WHAT WAS CREATED:")
        logger.info(f"  {len(folds)} cross-validation folds in: {base_output_dir}")

        if args.aggregation != 'none':
            logger.info("")
            logger.info("  Each fold tests a different generalization scenario:")
            for fold_name, _, _, test_samples in folds:
                logger.info(f"  - {fold_name}/: Can model detect lies in '{fold_name}' after training on other categories?")
        else:
            logger.info("")
            logger.info("  Each fold is a random split of the data:")
            for i in range(len(folds)):
                logger.info(f"  - fold_{i}/: Random subset for cross-validation")

        logger.info("")
        logger.info("  Each fold directory contains:")
        logger.info("  - train.jsonl: Training data for that fold")
        logger.info("  - val.jsonl: Validation data for that fold")
        if args.validation_split > 0:
            logger.info("  - test.jsonl: Test data for that fold")
        logger.info("  - metadata.json: Fold-specific statistics")

        # logger.info("")
        # logger.info("🚀 TO FINE-TUNE YOUR MODELS:")
        # logger.info(f"Run fine-tuning for each of the {len(folds)} folds:")
        #
        # for i, (fold_name, _, _, test_samples) in enumerate(folds):
        #     fold_dir = base_output_dir / fold_name
        #     print(f"\n# Fold {i + 1}/{len(folds)}: {fold_name}")
        #     if args.aggregation != 'none' and args.validation_split > 0:
        #         print(f"# Tests: Can model generalize to '{fold_name}' tasks?")
        #     print(f"python -m safetytooling.apis.finetuning.openai.run \\")
        #     print(f"    --model '{args.model}' \\")
        #     print(f"    --train_file {fold_dir}/train.jsonl \\")
        #     print(f"    --val_file {fold_dir}/val.jsonl \\")
        #     print(f"    --n_epochs 3")
        #     if test_samples:
        #         print(f"\n# After training, evaluate on test set:")
        #         print(f"# python eval.py --model <finetuned_model> --test_file {fold_dir}/test.jsonl")

        # print("\n💡 TIPS:")
        # print("  • Add --dry_run flag to estimate costs before running")
        # if args.validation_split > 0:
        #     print("  • Use val.jsonl for hyperparameter tuning and early stopping")
        #     print("  • Use test.jsonl for final evaluation only")
        # print("  • Each fold tests a different aspect of lie detection")
        # print("  • Results can be aggregated to measure overall performance")
        # print("\n" + "=" * 70)

    # Remove file handler and close it
    root_logger.removeHandler(file_handler)
    file_handler.close()


if __name__ == "__main__":
    main()