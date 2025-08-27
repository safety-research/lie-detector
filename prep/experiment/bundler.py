"""
Core bundling logic for preparing training data from prep datasets.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import datetime

from .processor import FormatProcessor


# Model name mapping to match training script expectations
MODEL_NAME_MAP = {
    'google/gemma-3-4b-it': 'openrouter_google_gemma-3-4b-it',
    'google/gemma-3-12b-it': 'openrouter_google_gemma-3-12b-it',
    'google/gemma-3-27b-it': 'openrouter_google_gemma-3-27b-it',
    'openai/gpt-4o': 'openai_gpt-4o',
    'openai/gpt-4o-mini': 'openai_gpt-4o-mini',
    'openai/gpt-3.5-turbo': 'openai_gpt-3.5-turbo',
    'anthropic/claude-3-opus': 'anthropic_claude-3-opus',
    'anthropic/claude-3.5-sonnet': 'anthropic_claude-3.5-sonnet',
}


class DataBundler:
    """Handles bundling of selected folds into train/eval sets."""
    
    def __init__(self):
        self.processor = FormatProcessor()
    
    def _load_fold_data(self, fold_dir: Path, fold_names: List[str], dataset_path: Path = None) -> List[Dict[str, Any]]:
        """
        Load data from specified folds.
        
        Args:
            fold_dir: Directory containing fold JSONL files (for folds_* structure)
            fold_names: List of fold names to load
            dataset_path: Dataset base path (for direct fold structure)
            
        Returns:
            List of data items from all specified folds
        """
        all_data = []
        
        for fold_name in fold_names:
            # Check if we're using direct fold structure (prep.dataset style)
            if dataset_path and (dataset_path / fold_name).is_dir():
                # Load from fold_name/train.jsonl
                fold_file = dataset_path / fold_name / "train.jsonl"
                if not fold_file.exists():
                    # Try test.jsonl if train doesn't exist
                    fold_file = dataset_path / fold_name / "test.jsonl"
                    if not fold_file.exists():
                        raise ValueError(f"No train.jsonl or test.jsonl found in fold: {fold_name}")
            else:
                # Original structure - load from fold_dir
                fold_file = fold_dir / f"{fold_name}.jsonl"
                if not fold_file.exists():
                    # Try with .json extension
                    fold_file = fold_dir / f"{fold_name}.json"
                    if not fold_file.exists():
                        raise ValueError(f"Fold file not found: {fold_name} in {fold_dir}")
            
            print(f"  Loading fold: {fold_file}")
            fold_data = []
            with open(fold_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            data_item = json.loads(line)
                            fold_data.append(data_item)
                        except json.JSONDecodeError as e:
                            print(f"    Error parsing line {line_num} in {fold_file}: {e}")
                            continue
            
            print(f"    Loaded {len(fold_data)} items from {fold_name}")
            all_data.extend(fold_data)
        
        return all_data
    
    def _balance_data(self, 
                      data_items: List[Dict[str, Any]], 
                      balance_strategy: str = "downsample",
                      format_type: str = "chat") -> List[Dict[str, Any]]:
        """
        Balance lies and truths in the dataset.
        
        Args:
            data_items: List of data items
            balance_strategy: "downsample" or "upsample"
            format_type: "chat" or "base_transcript"
            
        Returns:
            Balanced list of data items
        """
        # Separate lies and truths - handle both old and new formats
        lies = []
        truths = []
        
        for item in data_items:
            # Check for new format (meta field) first
            if 'meta' in item:
                did_lie = item['meta'].get('did_lie', False)
            else:
                # Old format - did_lie at root
                did_lie = item.get('did_lie', False)
            
            if did_lie:
                lies.append(item)
            else:
                truths.append(item)
        
        print(f"  Original data: {len(lies)} lies, {len(truths)} truths")
        
        # For chat format, filter out problematic examples BEFORE balancing
        if format_type == "chat":
            # Filter lies that can be converted to chat format
            filtered_lies = []
            for item in lies:
                if self.processor.create_chat_format_sample(item) is not None:
                    filtered_lies.append(item)
            
            # Filter truths that can be converted to chat format
            filtered_truths = []
            for item in truths:
                if self.processor.create_chat_format_sample(item) is not None:
                    filtered_truths.append(item)
            
            print(f"  After chat format filtering: {len(filtered_lies)} lies, {len(filtered_truths)} truths")
            
            # Use filtered data for balancing
            lies = filtered_lies
            truths = filtered_truths
        
        # Balance based on strategy
        if balance_strategy == "downsample":
            min_count = min(len(lies), len(truths))
            balanced_lies = random.sample(lies, min_count) if len(lies) > min_count else lies
            balanced_truths = random.sample(truths, min_count) if len(truths) > min_count else truths
        else:  # upsample
            max_count = max(len(lies), len(truths))
            balanced_lies = lies + random.choices(lies, k=max_count - len(lies)) if len(lies) < max_count else lies
            balanced_truths = truths + random.choices(truths, k=max_count - len(truths)) if len(truths) < max_count else truths
        
        balanced_data = balanced_lies + balanced_truths
        random.shuffle(balanced_data)
        
        print(f"  Balanced data: {len(balanced_lies)} lies, {len(balanced_truths)} truths")
        
        return balanced_data
    
    def _split_data(self, 
                    data_items: List[Dict[str, Any]], 
                    max_train_examples: Optional[int] = None,
                    max_eval_examples: Optional[int] = None,
                    val_split: Optional[float] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Split data into train/eval(/val) sets.
        
        Args:
            data_items: Balanced list of data items
            max_train_examples: Maximum training examples
            max_eval_examples: Maximum evaluation examples
            val_split: Optional validation split from training data
            
        Returns:
            Dictionary with 'train', 'eval', and optionally 'val' keys
        """
        # Separate lies and truths to ensure balanced splits - handle both formats
        lies = []
        truths = []
        
        for item in data_items:
            # Check for new format (meta field) first
            if 'meta' in item:
                did_lie = item['meta'].get('did_lie', False)
            else:
                # Old format - did_lie at root
                did_lie = item.get('did_lie', False)
            
            if did_lie:
                lies.append(item)
            else:
                truths.append(item)
        
        # Determine train size
        if max_train_examples is not None:
            train_lie_count = min(max_train_examples // 2, len(lies))
            train_truth_count = min(max_train_examples - train_lie_count, len(truths))
        else:
            # Use 80% for training
            train_lie_count = int(len(lies) * 0.8)
            train_truth_count = int(len(truths) * 0.8)
        
        # Create train split
        train_lies = lies[:train_lie_count]
        train_truths = truths[:train_truth_count]
        train_data = train_lies + train_truths
        random.shuffle(train_data)
        
        # Create eval split
        eval_lies = lies[train_lie_count:]
        eval_truths = truths[train_truth_count:]
        
        # Apply max_eval_examples if specified
        if max_eval_examples is not None:
            eval_lie_count = min(max_eval_examples // 2, len(eval_lies))
            eval_truth_count = min(max_eval_examples - eval_lie_count, len(eval_truths))
            eval_lies = eval_lies[:eval_lie_count]
            eval_truths = eval_truths[:eval_truth_count]
        
        # Ensure eval set has at least 128 samples
        total_eval = len(eval_lies) + len(eval_truths)
        if total_eval < 128:
            print(f"  Warning: Eval set only has {total_eval} samples, which is less than the minimum of 128.")
        
        eval_data = eval_lies + eval_truths
        random.shuffle(eval_data)
        
        splits = {
            'train': train_data,
            'eval': eval_data
        }
        
        # Handle validation split if requested
        if val_split is not None and val_split > 0:
            # Split training data into train/val
            val_size = int(len(train_data) * val_split)
            splits['val'] = train_data[:val_size]
            splits['train'] = train_data[val_size:]
        
        return splits
    
    def bundle_folds(self, 
                     dataset_path: str,
                     train_folds: List[str], 
                     eval_folds: List[str],
                     format_type: str = "chat",
                     max_train_examples: Optional[int] = None,
                     max_eval_examples: Optional[int] = None,
                     balance_strategy: str = "downsample",
                     val_split: Optional[float] = None,
                     seed: int = 42,
                     size_param: Optional[int] = None) -> Dict[str, Any]:
        """
        Bundle selected folds into train/eval sets.
        
        Args:
            dataset_path: Path to prep dataset
            train_folds: List of fold names for training
            eval_folds: List of fold names for evaluation
            format_type: "chat" or "base_transcript"
            max_train_examples: Maximum training examples
            max_eval_examples: Maximum evaluation examples
            balance_strategy: "downsample" or "upsample"
            val_split: Optional validation split from training data
            seed: Random seed
            
        Returns:
            Dictionary containing bundled data and statistics
        """
        random.seed(seed)
        
        # Need to determine which fold directory to use
        # This requires using FoldSelector to find the correct fold_dir
        from .selector import FoldSelector
        selector = FoldSelector(dataset_path)
        fold_info = selector.select_folds(train_folds, eval_folds)
        fold_dir = fold_info['fold_dir']
        
        print(f"\nBundling folds from: {fold_dir}")
        print(f"Train folds: {train_folds}")
        print(f"Eval folds: {eval_folds}")
        
        # Load data from folds
        # Check if we're using direct fold structure
        dataset_path_obj = Path(dataset_path)
        if fold_info.get('fold_type') == 'default':
            # Direct fold structure - pass dataset_path
            train_data = self._load_fold_data(fold_dir, train_folds, dataset_path=dataset_path_obj)
            eval_data = self._load_fold_data(fold_dir, eval_folds, dataset_path=dataset_path_obj)
        else:
            # Traditional folds_* structure
            train_data = self._load_fold_data(fold_dir, train_folds)
            eval_data = self._load_fold_data(fold_dir, eval_folds)
        
        # Balance data
        balanced_train = self._balance_data(train_data, balance_strategy, format_type)
        balanced_eval = self._balance_data(eval_data, balance_strategy, format_type)
        
        # Split data with size limits
        train_splits = self._split_data(
            balanced_train, 
            max_train_examples=max_train_examples,
            val_split=val_split
        )
        
        eval_splits = self._split_data(
            balanced_eval,
            max_eval_examples=max_eval_examples
        )
        
        # Use only the eval portion from eval_splits
        final_splits = {
            'train': train_splits['train'],
            'eval': eval_splits['eval']  # Use eval portion, not train
        }
        
        if 'val' in train_splits:
            final_splits['val'] = train_splits['val']
        
        # Convert to appropriate format
        formatted_data = {}
        for split_name, split_data in final_splits.items():
            formatted_examples = []
            for item in split_data:
                if format_type == "chat":
                    example = self.processor.create_chat_format_sample(item)
                else:
                    example = self.processor.create_base_transcript_sample(item)
                
                if example is not None:
                    formatted_examples.append(example)
            
            formatted_data[split_name] = formatted_examples
        
        # Log merge statistics if using chat format
        if format_type == "chat" and self.processor.merge_stats['total'] > 0:
            print(f"\nConsecutive assistant message merging:")
            print(f"  Total examples processed: {self.processor.merge_stats['total']}")
            print(f"  Examples with merges: {self.processor.merge_stats['merged']}")
            print(f"  Merge rate: {self.processor.merge_stats['merged'] / self.processor.merge_stats['total'] * 100:.1f}%")
        
        # Calculate statistics
        statistics = {}
        for split_name, examples in formatted_data.items():
            lies = sum(1 for ex in examples if ex['completion'] == 'A')
            truths = len(examples) - lies
            statistics[split_name] = {
                'total': len(examples),
                'lies': lies,
                'truths': truths
            }
        
        return {
            'data': formatted_data,
            'statistics': statistics,
            'bundled_folds': {
                'train': train_folds,
                'eval': eval_folds
            },
            'config': {
                'dataset_path': dataset_path,
                'format_type': format_type,
                'balance_strategy': balance_strategy,
                'seed': seed,
                'max_train_examples': max_train_examples,
                'max_eval_examples': max_eval_examples,
                'val_split': val_split,
                'size': size_param
            }
        }
    
    def save_bundle(self, bundle_data: Dict[str, Any], output_dir: str, model_name: str, bundler_instance=None):
        """
        Save bundled data matching the expected directory structure.
        
        Args:
            bundle_data: Bundle data from bundle_folds
            output_dir: Output directory path
            model_name: Model name (will be mapped to expected format)
        """
        output_path = Path(output_dir)
        
        # Map model name to expected format
        mapped_model_name = model_name.replace('-', '_')
        
        # Create experiment name
        train_folds = '_'.join(bundle_data['bundled_folds']['train'])
        format_type = bundle_data['config']['format_type']
        size = bundle_data['config'].get('size')
        
        # Include size in experiment name if specified
        if size:
            experiment_name = f"bundled_{train_folds}_{format_type}_size_{size}"
        else:
            experiment_name = f"bundled_{train_folds}_{format_type}"
        
        # Create directory structure
        model_dir = output_path / mapped_model_name
        experiment_dir = model_dir / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each split directly in experiment directory
        for split_name, examples in bundle_data['data'].items():
            # Use 'val' instead of 'eval' for validation set
            file_name = 'val.jsonl' if split_name == 'eval' else f"{split_name}.jsonl"
            split_file = experiment_dir / file_name
            
            with open(split_file, 'w', encoding='utf-8') as f:
                for example in examples:
                    f.write(json.dumps(example, ensure_ascii=False) + '\n')
            
            print(f"  Saved {split_name} data to: {split_file}")
        
        # Save metadata
        metadata = {
            'created_at': datetime.datetime.now().isoformat(),
            'bundled_folds': bundle_data['bundled_folds'],
            'statistics': bundle_data['statistics'],
            'config': bundle_data['config']
        }
        
        # Add merge statistics if available
        if bundle_data['config']['format_type'] == 'chat' and bundler_instance and hasattr(bundler_instance, 'processor'):
            metadata['merge_statistics'] = {
                'total_processed': bundler_instance.processor.merge_stats['total'],
                'examples_with_merges': bundler_instance.processor.merge_stats['merged'],
                'merge_rate': bundler_instance.processor.merge_stats['merged'] / bundler_instance.processor.merge_stats['total'] if bundler_instance.processor.merge_stats['total'] > 0 else 0
            }
        
        metadata_file = experiment_dir / 'metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        # Save training summary
        training_summary = {
            'input_directory': bundle_data['config']['dataset_path'],
            'output_directory': str(output_path),
            'model_name': mapped_model_name,
            'format_type': format_type,
            'seed': bundle_data['config']['seed'],
            'bundled_folds': bundle_data['bundled_folds'],
            'statistics': bundle_data['statistics']
        }
        
        summary_file = model_dir / 'training_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(training_summary, f, indent=2)
        
        print(f"\nBundle saved successfully!")
        print(f"Model directory: {model_dir}")
        print(f"Experiment directory: {experiment_dir}")