"""
CLI entry point for the prep.bundle module.

Usage:
    python -m prep.bundle \
        --dataset .data/openai/gpt_4o \
        --model google/gemma-3-4b-it \
        --format chat \
        --train games \
        --train ascii \
        --output ./bundled_data
"""

import argparse
import sys
from pathlib import Path
import json

from .bundler import DataBundler
from .selector import FoldSelector


def main():
    parser = argparse.ArgumentParser(
        description="Bundle prep dataset folds for training",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to prep dataset directory"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., google/gemma-3-4b-it)"
    )
    
    parser.add_argument(
        "--format",
        type=str,
        choices=["chat", "base_transcript"],
        default="chat",
        help="Output format (default: chat)"
    )
    
    parser.add_argument(
        "--train",
        type=str,
        action="append",
        required=True,
        help="Training fold names (can be specified multiple times)"
    )
    
    parser.add_argument(
        "--eval",
        type=str,
        action="append",
        help="Evaluation fold names (optional, uses remaining folds if not specified)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="./experiment",
        help="Output directory path (default: ./experiment)"
    )
    
    parser.add_argument(
        "--max-train-examples",
        type=int,
        help="Maximum training examples (optional)"
    )
    
    parser.add_argument(
        "--max-eval-examples",
        type=int,
        help="Maximum evaluation examples (optional)"
    )
    
    parser.add_argument(
        "--size",
        type=int,
        help="Cap both train and val datasets to this size (overrides individual max settings)"
    )
    
    parser.add_argument(
        "--balance",
        type=str,
        choices=["downsample", "upsample"],
        default="downsample",
        help="Balance strategy (default: downsample)"
    )
    
    parser.add_argument(
        "--val-split",
        type=float,
        help="Validation split from training data (0.0-0.3)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.val_split is not None and not 0.0 <= args.val_split <= 0.3:
        parser.error("--val-split must be between 0.0 and 0.3")
    
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        parser.error(f"Dataset path does not exist: {dataset_path}")
    
    # Select folds
    selector = FoldSelector(args.dataset)
    
    try:
        available_folds = selector.discover_folds()
        print(f"Available folds: {json.dumps(available_folds, indent=2)}")
        
        fold_selection = selector.select_folds(
            train_folds=args.train,
            eval_folds=args.eval
        )
        
        print(f"\nSelected folds:")
        print(f"  Train: {fold_selection['train']}")
        print(f"  Eval: {fold_selection['eval']}")
        
    except ValueError as e:
        parser.error(str(e))
    
    # Bundle data
    bundler = DataBundler()
    
    try:
        print(f"\nBundling data...")
        
        # Handle size parameter - it overrides individual max settings
        max_train = args.size if args.size else args.max_train_examples
        max_eval = args.size if args.size else args.max_eval_examples
        
        # Warn if eval size will be less than 128
        if max_eval is not None and max_eval < 128:
            print(f"  WARNING: Eval size of {max_eval} is less than the recommended minimum of 128.")
            print(f"           Consider using --size 128 or higher, or --max-eval-examples 128 or higher.")
        
        bundle_data = bundler.bundle_folds(
            dataset_path=args.dataset,
            train_folds=fold_selection['train'],
            eval_folds=fold_selection['eval'],
            format_type=args.format,
            max_train_examples=max_train,
            max_eval_examples=max_eval,
            balance_strategy=args.balance,
            val_split=args.val_split,
            seed=args.seed,
            size_param=args.size  # Pass the original size parameter for naming
        )
        
        # Save bundle
        bundler.save_bundle(
            bundle_data=bundle_data,
            output_dir=args.output,
            model_name=args.model,
            bundler_instance=bundler
        )
        
        print(f"\nBundle saved to: {args.output}")
        print(f"Training samples: {bundle_data['statistics']['train']['total']}")
        print(f"Evaluation samples: {bundle_data['statistics']['eval']['total']}")
        
        # Warn if eval set ended up being less than 128
        eval_total = bundle_data['statistics']['eval']['total']
        if eval_total < 128:
            print(f"\n⚠️  WARNING: Evaluation set has only {eval_total} samples, which is less than the recommended minimum of 128.")
            print(f"    This may cause issues with evaluation metrics.")
        
        if 'val' in bundle_data['statistics']:
            print(f"Validation samples: {bundle_data['statistics']['val']['total']}")
        
    except Exception as e:
        print(f"Error bundling data: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()