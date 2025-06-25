#!/usr/bin/env python3
"""
Fine-tune a model for lie detection using the sycophancy dataset.

This script uses the safety-tooling API to fine-tune a model (likely Llama 3.1)
to detect lies in responses to questions.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add safety-tooling to path
safety_tooling_path = Path("/workspace/LieDetector/safety-tooling")
sys.path.insert(0, str(safety_tooling_path))

from safetytooling.apis.finetuning.together.run import main as together_main, TogetherFTConfig


async def finetune_lie_detector(
    model: str = "meta-llama/Llama-3.1-8B-Instruct",
    train_file: str = "/workspace/LieDetector/finetuning/data/lie_detection_train.jsonl",
    val_file: str = "/workspace/LieDetector/finetuning/data/lie_detection_val.jsonl",
    n_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    wandb_project: str = "lie-detector-finetuning",
    wandb_entity: str = None,
    save_folder: str = "/workspace/LieDetector/finetuning/models",
    suffix: str = "lie-detector",
    lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    dry_run: bool = False
) -> None:
    """
    Fine-tune a model for lie detection.
    
    Args:
        model: Model to fine-tune (default: Llama 3.1 8B Instruct)
        train_file: Path to training data JSONL file
        val_file: Path to validation data JSONL file
        n_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for training
        wandb_project: Weights & Biases project name
        wandb_entity: Weights & Biases entity/username
        save_folder: Directory to save the fine-tuned model
        suffix: Suffix to append to model name
        lora: Whether to use LoRA for efficient fine-tuning
        lora_r: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        dry_run: If True, only validate data without launching job
    """
    
    # Check if Together API key is set
    if not os.environ.get("TOGETHER_API_KEY"):
        raise ValueError("TOGETHER_API_KEY environment variable must be set")
    
    # Check if files exist
    train_path = Path(train_file)
    val_path = Path(val_file)
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation file not found: {val_file}")
    
    # Create configuration
    config = TogetherFTConfig(
        train_file=train_path,
        val_file=val_path,
        model=model,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        lora=lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        suffix=suffix,
        wandb_project_name=wandb_project,
        wandb_entity=wandb_entity,
        tags=("lie-detection", "sycophancy", "safety"),
        save_folder=save_folder,
        save_config=True,
        save_model=True,
        dry_run=dry_run,
        logging_level="info"
    )
    
    print("Fine-tuning configuration:")
    print(f"  Model: {model}")
    print(f"  Training file: {train_file}")
    print(f"  Validation file: {val_file}")
    print(f"  Epochs: {n_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  LoRA: {lora}")
    if lora:
        print(f"  LoRA rank: {lora_r}")
        print(f"  LoRA alpha: {lora_alpha}")
        print(f"  LoRA dropout: {lora_dropout}")
    print(f"  Save folder: {save_folder}")
    print(f"  Dry run: {dry_run}")
    
    # Run fine-tuning
    print("\nStarting fine-tuning...")
    try:
        result = await together_main(config)
        print(f"\nFine-tuning completed successfully!")
        print(f"Model ID: {result.id}")
        if hasattr(result, 'output_name') and result.output_name:
            print(f"Output model: {result.output_name}")
    except Exception as e:
        print(f"Fine-tuning failed: {e}")
        raise


def main():
    """Main function to run lie detector fine-tuning."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune a model for lie detection")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct-Reference", 
                       help="Model to fine-tune")
    parser.add_argument("--train-file", 
                       default="/workspace/LieDetector/finetuning/data/lie_detection_train.jsonl",
                       help="Path to training data")
    parser.add_argument("--val-file", 
                       default="/workspace/LieDetector/finetuning/data/lie_detection_val.jsonl",
                       help="Path to validation data")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--wandb-project", default="lie-detector-finetuning",
                       help="Weights & Biases project name")
    parser.add_argument("--wandb-entity", help="Weights & Biases entity/username")
    parser.add_argument("--save-folder", 
                       default="/workspace/LieDetector/finetuning/models",
                       help="Directory to save fine-tuned model")
    parser.add_argument("--suffix", default="lie-detector", help="Model name suffix")
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA fine-tuning")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--dry-run", action="store_true", help="Validate data only")
    
    args = parser.parse_args()
    
    # Run fine-tuning
    asyncio.run(finetune_lie_detector(
        model=args.model,
        train_file=args.train_file,
        val_file=args.val_file,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        save_folder=args.save_folder,
        suffix=args.suffix,
        lora=not args.no_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        dry_run=args.dry_run
    ))


if __name__ == "__main__":
    main() 