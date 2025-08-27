import argparse
import hashlib
import json
import logging
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from common.generalization_mappings import generalization_map_1, generalization_map_2, generalization_map_3
from prep.download import S3DataDownloader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def merge_consecutive_assistant_messages(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Merge consecutive assistant messages into a single message.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys

    Returns:
        Tuple of (merged messages list, whether any merging occurred)
    """
    if not messages:
        return messages, False

    merged_messages = []
    current_assistant_content = ""
    merge_occurred = False
    consecutive_count = 0

    for message in messages:
        if message.get("role") == "assistant":
            # Accumulate assistant content
            current_assistant_content += message.get("content", "")
            consecutive_count += 1
            if consecutive_count > 1:
                merge_occurred = True
        else:
            # If we have accumulated assistant content, add it as a single message
            if current_assistant_content:
                merged_messages.append({
                    "role": "assistant",
                    "content": current_assistant_content
                })
                current_assistant_content = ""
                consecutive_count = 0

            # Add the non-assistant message
            merged_messages.append(message)

    # Don't forget to add any remaining assistant content at the end
    if current_assistant_content:
        merged_messages.append({
            "role": "assistant",
            "content": current_assistant_content
        })

    return merged_messages, merge_occurred


class LieDetectionDataProcessor:
    """Processes lie detection samples for fine-tuning."""

    def __init__(self, samples: List[Dict]):
        self.samples = samples
        self.generalization_maps = {
            'motivation': generalization_map_1,
            'knowledge': generalization_map_2,
            'task-group': generalization_map_3
        }
        # Track merge statistics
        self.merge_stats = {
            'total': 0,
            'merged': 0
        }

    def deduplicate_samples(self) -> List[Dict]:
        """Deduplicate samples based on trace content and did_lie value."""
        seen_hashes = set()
        unique_samples = []

        for sample in self.samples:
            # Extract trace content and did_lie value
            trace_content = []
            if 'trace' in sample:
                # Convert trace to a consistent string representation
                for turn in sample['trace']:
                    trace_content.append(f"{turn.get('role', '')}: {turn.get('content', '')}")

            # Combine trace content and did_lie value for hashing
            trace_str = "\n".join(trace_content)
            did_lie = sample.get('did_lie', None)

            # Create a unique identifier based on trace and did_lie
            content_to_hash = f"{trace_str}|did_lie:{did_lie}"
            sample_hash = hashlib.md5(content_to_hash.encode()).hexdigest()

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

    def process_trace_messages(self, trace: List[Dict]) -> List[Dict]:
        """
        Process trace messages to ensure proper formatting and merge consecutive assistant messages.

        Args:
            trace: List of message dictionaries from the sample trace

        Returns:
            Processed and merged messages
        """
        # First, ensure all messages have the correct format
        formatted_messages = []
        for turn in trace:
            role = turn.get('role', 'unknown')
            content = turn.get('content', '')

            # Normalize role names
            if role.lower() == 'system':
                formatted_messages.append({'role': 'system', 'content': content})
            elif role.lower() == 'user':
                formatted_messages.append({'role': 'user', 'content': content})
            elif role.lower() == 'assistant':
                formatted_messages.append({'role': 'assistant', 'content': content})
            else:
                # Handle unknown roles as user messages
                formatted_messages.append({'role': 'user', 'content': content})

        # Now merge consecutive assistant messages
        merged_messages, had_merge = merge_consecutive_assistant_messages(formatted_messages)

        # Update statistics
        self.merge_stats['total'] += 1
        if had_merge:
            self.merge_stats['merged'] += 1

        return merged_messages

    def create_folds(self, categorized_samples: Dict[str, List[Dict]],
                     size: Optional[int] = None,
                     validation_split: float = 0.2) -> List[Tuple[str, List[Dict], List[Dict], Optional[List[Dict]]]]:
        """
        Create train/validation folds based on categories.

        Each fold contains samples from only one category.

        Args:
            categorized_samples: Dict of category -> samples
            size: Optional max training size
            validation_split: Fraction of data for validation (default 0.2)

        Returns:
            List of (fold_name, train, val, test) tuples
            test will always be None (no test set created)
        """
        categories = list(categorized_samples.keys())

        if len(categories) == 1:
            # Single category - standard split
            all_samples = categorized_samples[categories[0]]
            shuffled = all_samples.copy()
            random.shuffle(shuffled)

            # Two-way split: 80/20 by default
            val_size = int(validation_split * len(shuffled))
            train_size = len(shuffled) - val_size

            train = shuffled[:train_size]
            val = shuffled[train_size:]

            return [("all", train, val, None)]

        # Multiple categories - create fold for each category
        folds = []

        for fold_category in categories:
            # Get samples ONLY from this category
            category_samples = categorized_samples[fold_category]

            # Shuffle the category samples
            random.shuffle(category_samples)

            # Calculate split sizes for this category
            total_samples = len(category_samples)
            val_size = int(validation_split * total_samples)
            train_size = total_samples - val_size

            # Split the category data
            val_samples = category_samples[:val_size]
            train_samples = category_samples[val_size:]

            # Apply size limit if specified (only to training data)
            if size and len(train_samples) > size:
                train_samples = random.sample(train_samples, size)

            # Log the fold information
            logger.info(f"Fold '{fold_category}':")
            logger.info(f"  - Train: {len(train_samples)} samples from '{fold_category}' category only")
            logger.info(f"  - Val: {len(val_samples)} samples from '{fold_category}' category only")
            logger.info(f"  - This fold contains ONLY '{fold_category}' samples")

            folds.append((fold_category, train_samples, val_samples, None))

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


def save_training_files(train_data: List[Dict], val_data: List[Dict],
                        output_dir: Path,
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

    # Save metadata
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(fold_info, f, indent=2)

    logger.info(f"Saved {len(train_data)} training samples to {train_file}")
    logger.info(f"Saved {len(val_data)} validation samples to {val_file}")
    logger.info(f"Saved fold metadata to {metadata_file}")


def verify_dataset(base_dir: Path, aggregation: str = 'task-group'):
    """
    Verify dataset by analyzing the distribution of did_lie/didn't lie for each domain.

    Args:
        base_dir: Base directory containing the dataset files
        aggregation: Aggregation strategy ('motivation', 'knowledge', 'task-group', 'none')
    """
    logger.info("=" * 70)
    logger.info("DATASET VERIFICATION - LIE DISTRIBUTION ANALYSIS")
    logger.info("=" * 70)
    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Aggregation: {aggregation}")
    logger.info("")

    # Get the generalization map
    generalization_maps = {
        'motivation': generalization_map_1,
        'knowledge': generalization_map_2,
        'task-group': generalization_map_3
    }

    # Overall statistics
    overall_stats = {
        'total_samples': 0,
        'total_lies': 0,
        'total_truths': 0,
        'by_domain': {},
        'by_task': defaultdict(lambda: {'lies': 0, 'truths': 0, 'total': 0}),
        'consecutive_merges': 0,
        'samples_with_merges': 0
    }

    # Find all train/val files
    dataset_files = list(base_dir.glob("**/train.jsonl")) + list(base_dir.glob("**/val.jsonl"))

    if not dataset_files:
        logger.error(f"No dataset files found in {base_dir}")
        return

    logger.info(f"Found {len(dataset_files)} dataset files")
    logger.info("")

    # Process each file
    for file_path in dataset_files:
        file_type = "train" if file_path.name == "train.jsonl" else "val"
        domain = file_path.parent.name if file_path.parent != base_dir else "root"

        if domain not in overall_stats['by_domain']:
            overall_stats['by_domain'][domain] = {
                'train': {'lies': 0, 'truths': 0, 'total': 0},
                'val': {'lies': 0, 'truths': 0, 'total': 0},
                'combined': {'lies': 0, 'truths': 0, 'total': 0},
                'by_task': defaultdict(lambda: {'lies': 0, 'truths': 0, 'total': 0})
            }

        # Read and analyze file
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    sample = json.loads(line)
                    meta = sample.get('meta', {})
                    did_lie = meta.get('did_lie', False)
                    task = meta.get('task', 'unknown')

                    # Check if messages were merged
                    if meta.get('had_consecutive_merge', False):
                        overall_stats['samples_with_merges'] += 1

                    # Update overall stats
                    overall_stats['total_samples'] += 1
                    if did_lie:
                        overall_stats['total_lies'] += 1
                    else:
                        overall_stats['total_truths'] += 1

                    # Update domain stats
                    overall_stats['by_domain'][domain][file_type]['total'] += 1
                    overall_stats['by_domain'][domain]['combined']['total'] += 1

                    if did_lie:
                        overall_stats['by_domain'][domain][file_type]['lies'] += 1
                        overall_stats['by_domain'][domain]['combined']['lies'] += 1
                    else:
                        overall_stats['by_domain'][domain][file_type]['truths'] += 1
                        overall_stats['by_domain'][domain]['combined']['truths'] += 1

                    # Update task stats
                    overall_stats['by_task'][task]['total'] += 1
                    overall_stats['by_domain'][domain]['by_task'][task]['total'] += 1

                    if did_lie:
                        overall_stats['by_task'][task]['lies'] += 1
                        overall_stats['by_domain'][domain]['by_task'][task]['lies'] += 1
                    else:
                        overall_stats['by_task'][task]['truths'] += 1
                        overall_stats['by_domain'][domain]['by_task'][task]['truths'] += 1

                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse line in {file_path}")
                except Exception as e:
                    logger.warning(f"Error processing sample in {file_path}: {e}")

    # Print results
    logger.info("📊 OVERALL STATISTICS:")
    logger.info(f"  Total samples: {overall_stats['total_samples']:,}")
    logger.info(
        f"  Total lies: {overall_stats['total_lies']:,} ({overall_stats['total_lies'] / overall_stats['total_samples'] * 100:.1f}%)")
    logger.info(
        f"  Total truths: {overall_stats['total_truths']:,} ({overall_stats['total_truths'] / overall_stats['total_samples'] * 100:.1f}%)")
    if overall_stats['samples_with_merges'] > 0:
        logger.info(
            f"  Samples with consecutive assistant merges: {overall_stats['samples_with_merges']:,} ({overall_stats['samples_with_merges'] / overall_stats['total_samples'] * 100:.1f}%)")
    logger.info("")

    # Print domain statistics
    logger.info("📁 STATISTICS BY DOMAIN:")
    for domain, stats in sorted(overall_stats['by_domain'].items()):
        logger.info(f"\n  Domain: {domain}")
        logger.info(f"  " + "-" * 50)

        # Combined stats
        combined = stats['combined']
        if combined['total'] > 0:
            logger.info(f"  Combined: {combined['total']:,} samples")
            logger.info(f"    - Lies: {combined['lies']:,} ({combined['lies'] / combined['total'] * 100:.1f}%)")
            logger.info(f"    - Truths: {combined['truths']:,} ({combined['truths'] / combined['total'] * 100:.1f}%)")

        # Train/Val breakdown
        for split in ['train', 'val']:
            split_stats = stats[split]
            if split_stats['total'] > 0:
                logger.info(f"  {split.capitalize()}: {split_stats['total']:,} samples")
                logger.info(
                    f"    - Lies: {split_stats['lies']:,} ({split_stats['lies'] / split_stats['total'] * 100:.1f}%)")
                logger.info(
                    f"    - Truths: {split_stats['truths']:,} ({split_stats['truths'] / split_stats['total'] * 100:.1f}%)")

        # Top tasks in this domain
        if stats['by_task']:
            logger.info(f"\n  Top tasks in {domain}:")
            sorted_tasks = sorted(stats['by_task'].items(),
                                  key=lambda x: x[1]['total'],
                                  reverse=True)[:5]
            for task, task_stats in sorted_tasks:
                logger.info(f"    • {task}: {task_stats['total']:,} samples "
                            f"({task_stats['lies']:,} lies, {task_stats['truths']:,} truths)")

    # Print task statistics
    logger.info("\n📋 TOP TASKS OVERALL:")
    sorted_tasks = sorted(overall_stats['by_task'].items(),
                          key=lambda x: x[1]['total'],
                          reverse=True)[:20]

    for task, stats in sorted_tasks:
        lie_pct = stats['lies'] / stats['total'] * 100 if stats['total'] > 0 else 0
        logger.info(f"  • {task}: {stats['total']:,} samples "
                    f"({stats['lies']:,} lies [{lie_pct:.1f}%], "
                    f"{stats['truths']:,} truths [{100 - lie_pct:.1f}%])")

    # Save verification report
    report_file = base_dir / "verification_report.json"
    with open(report_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'base_dir': str(base_dir),
            'aggregation': aggregation,
            'overall': {
                'total_samples': overall_stats['total_samples'],
                'total_lies': overall_stats['total_lies'],
                'total_truths': overall_stats['total_truths'],
                'lie_percentage': overall_stats['total_lies'] / overall_stats['total_samples'] * 100 if overall_stats[
                                                                                                            'total_samples'] > 0 else 0,
                'samples_with_merges': overall_stats['samples_with_merges']
            },
            'by_domain': dict(overall_stats['by_domain']),
            'by_task': dict(overall_stats['by_task'])
        }, f, indent=2)

    logger.info(f"\n✅ Verification report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(description='Prepare lie detection data for fine-tuning')
    parser.add_argument('--model', help='Model identifier (e.g., openai/gpt-4o)', default='google/gemma-3-12b-it')
    parser.add_argument('--aggregation',
                        choices=['motivation', 'knowledge', 'task-group', 'none'],
                        default='task-group', help='Aggregation strategy for categorizing tasks')
    parser.add_argument('--balance', choices=['downsample', 'upsample', 'none'], default='downsample',
                        help='Strategy for balancing lie/truth samples')
    parser.add_argument('--validation-split', type=float, default=0.15,
                        help='Fraction of training data for validation (0.0-0.5). '
                             'If 0.0, uses OOD data as validation with no test set.')
    parser.add_argument('--size', type=int, help='Size of training data (optional)', default=None)
    parser.add_argument('--output-dir', default='.data', help='Base output directory for training files')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    # Hugging Face upload options
    parser.add_argument('--upload-to-hf', action='store_true', default=True,
                        help='Upload dataset to Hugging Face after creation')
    parser.add_argument('--hf-repo', default='Noddybear/lies', help='Hugging Face repository ID')
    parser.add_argument('--hf-token', help='Hugging Face API token (or set HF_TOKEN env var)')
    parser.add_argument('--hf-private', action='store_true', help='Make uploaded dataset private')

    # Add verify mode
    parser.add_argument('--verify', action='store_true', help='Run verification mode to analyze lie distribution')

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

    # If verify mode, run verification and exit
    if args.verify:
        verify_dataset(base_output_dir, args.aggregation)
        return

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
    logger.info(f"  Balance: {args.balance}")
    logger.info(f"  Validation split: {args.validation_split}")
    logger.info(f"  Size: {args.size}")
    logger.info(f"  Output directory: {base_output_dir}")
    logger.info(f"  Upload to HF: {args.upload_to_hf}")

    # Explain the strategy
    logger.info("")
    logger.info("📋 DATASET STRATEGY:")
    if args.aggregation != 'none':
        logger.info(f"  • Using {args.aggregation} to group tasks into categories")
        logger.info("  • Creating a single split")
        logger.info("    → 80/20 train/val split")
    else:
        logger.info("  • No task categorization - treating all samples equally")
        logger.info("  • Creating a single split")
        logger.info("    → 80/20 train/val split")

    if args.balance != 'none':
        logger.info(f"  • Balancing lies vs truths using {args.balance} strategy")
        if args.balance == 'downsample':
            logger.info("    → Reducing majority class to match minority class size")
        else:
            logger.info("    → Duplicating minority class samples to match majority class")

    if args.size:
        logger.info(f"  • Limiting training data to {args.size} samples per fold")

    logger.info("  • Merging consecutive assistant messages in traces")

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
    folds = []

    if args.aggregation != 'none':
        logger.info(f"  Creating per-category folds...")
        logger.info(f"  NOTE: Each fold contains ONLY samples from that category")
        folds_with_metadata = processor.create_folds(categorized, args.size, args.validation_split)

        # Extract folds and metadata
        fold_metadata_dict = {}
        for fold_data in folds_with_metadata:
            if len(fold_data) == 4:  # Format with metadata
                fold_name, train, val, _ = fold_data
                folds.append((fold_name, train, val, None))
            else:  # Old format
                folds.append(fold_data)
    else:
        # Single split or numeric folds
        all_samples = []
        for samples in categorized.values():
            all_samples.extend(samples)

        if args.size and len(all_samples) > args.size:
            all_samples = random.sample(all_samples, args.size)

        # Single fold (no subdirectory)
        logger.info(f"  Creating single 80/20 train/val split...")
        splits = processor._create_kfold_splits(all_samples, None, args.validation_split)
        folds = [("", splits[0][0], splits[0][1], None)]  # Empty name for single fold

    # Create master summary
    master_summary = {
        "model": args.model,
        "aggregation": args.aggregation,
        "validation_split": args.validation_split,
        "balance": args.balance,
        "requested_size": args.size,
        "total_samples": len(unique_samples),
        "num_folds": len(folds),
        "has_test_set": False,  # No test sets anymore
        "timestamp": datetime.now().isoformat(),
        "seed": args.seed,
        "fold_directories": [],
        "message_processing": {
            "consecutive_assistant_merging": True,
            "total_samples_processed": 0,
            "samples_with_merges": 0
        }
    }

    # Track created directories for HF upload
    created_directories = []

    # Convert to training format and save
    logger.info("")
    logger.info("💾 SAVING DATASETS:")

    for fold_name, train_samples, val_samples, test_samples in folds:
        # Need to preserve meta information for HF upload
        train_data = []
        val_data = []

        # Convert training samples while preserving metadata and merging messages
        for sample in train_samples:
            # Process and merge consecutive assistant messages
            merged_messages = processor.process_trace_messages(sample['trace'])

            # Check if merging occurred for this sample
            had_merge = len(merged_messages) != len(sample['trace'])
            if had_merge:
                master_summary["message_processing"]["samples_with_merges"] += 1
            master_summary["message_processing"]["total_samples_processed"] += 1

            train_data.append({
                "messages": merged_messages,
                "meta": {
                    "did_lie": sample['did_lie'],
                    "task": sample.get('task', 'unknown'),
                    "sample_id": sample.get('sample_id', 'unknown'),
                    "task_id": sample.get('task_id', 'unknown'),
                    "category": sample.get('category', 'unknown'),
                    "had_consecutive_merge": had_merge  # Track if merging occurred
                }
            })

        # Convert validation samples
        for sample in val_samples:
            # Process and merge consecutive assistant messages
            merged_messages = processor.process_trace_messages(sample['trace'])

            # Check if merging occurred for this sample
            had_merge = len(merged_messages) != len(sample['trace'])
            if had_merge:
                master_summary["message_processing"]["samples_with_merges"] += 1
            master_summary["message_processing"]["total_samples_processed"] += 1

            val_data.append({
                "messages": merged_messages,
                "meta": {
                    "did_lie": sample['did_lie'],
                    "task": sample.get('task', 'unknown'),
                    "sample_id": sample.get('sample_id', 'unknown'),
                    "task_id": sample.get('task_id', 'unknown'),
                    "category": sample.get('category', 'unknown'),
                    "had_consecutive_merge": had_merge
                }
            })

        # Determine output directory
        if fold_name:  # Multi-fold case
            output_dir = base_output_dir / fold_name
            master_summary["fold_directories"].append(fold_name)
            created_directories.append(output_dir)
            logger.info("")
            logger.info(f"  📂 Creating fold: {output_dir}")

            if args.aggregation != 'none':
                logger.info(f"     Purpose: Training on '{fold_name}' category only")
                logger.info(f"     Training data: 80% of '{fold_name}' samples")
                logger.info(f"     Validation data: 20% of '{fold_name}' samples")
                logger.info(f"     NOTE: This fold contains ONLY '{fold_name}' samples")
        else:  # Single fold case
            output_dir = base_output_dir
            created_directories.append(output_dir)
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
            "train_lies": len([s for s in train_samples if s.get('did_lie', False)]),
            "train_truths": len([s for s in train_samples if not s.get('did_lie', False)]),
            "val_lies": len([s for s in val_samples if s.get('did_lie', False)]),
            "val_truths": len([s for s in val_samples if not s.get('did_lie', False)]),
            "timestamp": datetime.now().isoformat(),
            "seed": args.seed,
            "split_type": "train/val",
            "message_processing": {
                "consecutive_assistant_merging": True
            }
        }

        # If using aggregation, add category info
        if args.aggregation != 'none' and fold_name:
            fold_info["fold_category"] = fold_name
            fold_info["contains_categories"] = [fold_name]  # Only contains this category
            fold_info["category_only"] = True

        save_training_files(train_data, val_data, output_dir, fold_info)

        # Print what was saved
        logger.info(
            f"     ✓ train.jsonl: {len(train_data)} samples ({fold_info['train_lies']} lies, {fold_info['train_truths']} truths)")
        logger.info(
            f"     ✓ val.jsonl: {len(val_data)} samples ({fold_info['val_lies']} lies, {fold_info['val_truths']} truths)")
        logger.info(f"     ✓ metadata.json: Fold configuration and statistics")

    # Log merge statistics
    if processor.merge_stats['total'] > 0:
        merge_rate = processor.merge_stats['merged'] / processor.merge_stats['total'] * 100
        logger.info("")
        logger.info("🔀 CONSECUTIVE ASSISTANT MESSAGE MERGING:")
        logger.info(f"  Total samples processed: {processor.merge_stats['total']}")
        logger.info(f"  Samples with merges: {processor.merge_stats['merged']}")
        logger.info(f"  Merge rate: {merge_rate:.1f}%")

        # Update master summary with final merge stats
        master_summary["message_processing"]["merge_rate"] = merge_rate

    # Save master summary
    summary_file = base_output_dir / "dataset_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(master_summary, f, indent=2)

    logger.info(f"Master summary saved to: {summary_file}")
    logger.info("     → Contains overview of all folds and dataset creation parameters")

    # Upload to Hugging Face if requested
    if args.upload_to_hf:
        logger.info("\n" + "=" * 70)
        logger.info("📤 UPLOADING TO HUGGING FACE")
        logger.info("=" * 70)

        try:
            from prep.hf_upload import HuggingFaceUploader

            uploader = HuggingFaceUploader(repo_id=args.hf_repo, token=args.hf_token)
            uploader.upload_all_folds(base_output_dir)

        except ImportError:
            logger.error("Could not import HuggingFaceUploader. Make sure to install required dependencies:")
            logger.error("pip install datasets huggingface_hub pandas")
        except Exception as e:
            logger.error(f"Failed to upload to Hugging Face: {e}")

    # Print instructions for fine-tuning
    logger.info("\n" + "=" * 70)
    logger.info("✅ DATASET PREPARATION COMPLETE!")
    logger.info("=" * 70)

    if len(folds) == 1 and not folds[0][0]:  # Single fold
        logger.info("")
        logger.info("📌 WHAT WAS CREATED:")
        logger.info(f"  A single train/validation split in: {base_output_dir}")
        logger.info("  - train.jsonl: Training data (with merged consecutive assistant messages)")
        logger.info("  - val.jsonl: Validation data (with merged consecutive assistant messages)")
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
    else:
        logger.info("")
        logger.info("📌 WHAT WAS CREATED:")
        logger.info(f"  {len(folds)} cross-validation folds in: {base_output_dir}")

        if args.aggregation != 'none':
            logger.info("")
            logger.info("  Each fold tests a different generalization scenario:")
            for fold_name, _, _, test_samples in folds:
                logger.info(
                    f"  - {fold_name}/: Can model detect lies other categories after training on '{fold_name}'?")
        else:
            logger.info("")
            logger.info("  Each fold is a random split of the data:")
            for i in range(len(folds)):
                logger.info(f"  - fold_{i}/: Random subset for cross-validation")

        logger.info("")
        logger.info("  Each fold directory contains:")
        logger.info("  - train.jsonl: Training data for that fold (with merged consecutive assistant messages)")
        logger.info("  - val.jsonl: Validation data for that fold (with merged consecutive assistant messages)")
        logger.info("  - metadata.json: Fold-specific statistics")

    # Remove file handler and close it
    root_logger.removeHandler(file_handler)
    file_handler.close()


if __name__ == "__main__":
    main()