#!/usr/bin/env python3
"""
Unified SFT (Supervised Fine-Tuning) script for lie detection models.

Usage:
    python -m sft --model google/gemma-3-12b-it --experiment bundled_games_ascii_chat --size 128
"""

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import torch
import wandb
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# HuggingFace token
HF_TOKEN = os.getenv("HF_TOKEN")


class LieDetectionSFT:
    """Main class for supervised fine-tuning of lie detection models."""

    # Model mappings for different platforms
    MODEL_MAPPINGS = {
        'google/gemma-3-4b-it': 'google/gemma-3-4b-it',
        'google/gemma-3-12b-it': 'google/gemma-3-12b-it',
        'google/gemma-3-27b-it': 'google/gemma-3-27b-it',
    }

    def __init__(self, args):
        self.args = args
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.setup_directories()
        self.authenticate_huggingface()

    def authenticate_huggingface(self):
        """Authenticate with HuggingFace using the provided token."""
        token = os.environ.get('HF_TOKEN', HF_TOKEN)

        try:
            login(token=token)
            logger.info("Successfully authenticated with HuggingFace")
        except Exception as e:
            logger.warning(f"Failed to authenticate with HuggingFace: {e}")
            logger.warning("Continuing without authentication - this may fail for gated models")

    def setup_directories(self):
        """Create necessary directories for the run."""
        self.run_dir = Path(f"runs/sft_{self.timestamp}")
        self.data_dir = self.run_dir / "data"
        self.model_dir = self.run_dir / "models"

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)

        logger.info(f"Created run directory: {self.run_dir}")

    def discover_experiments(self, data_path: Path) -> List[str]:
        """Discover available experiments in the bundled dataset."""
        experiments = []

        # Look for directories that start with "bundled_"
        for item in data_path.iterdir():
            if item.is_dir() and item.name.startswith("bundled_"):
                # Check if it has the required files
                if (item / "train.jsonl").exists() and (item / "val.jsonl").exists():
                    experiments.append(item.name)

        return sorted(experiments)

    def load_bundled_data(self, experiment_path: Path) -> Tuple[List[Dict], List[Dict], Dict]:
        """Load data from a bundled experiment."""
        train_data = []
        eval_data = []

        # Load training data
        train_file = experiment_path / "train.jsonl"
        if train_file.exists():
            with open(train_file, 'r') as f:
                for line in f:
                    if line.strip():
                        train_data.append(json.loads(line))
            logger.info(f"Loaded {len(train_data)} training samples from {train_file}")

        # Load evaluation data (named val.jsonl in bundler output)
        eval_file = experiment_path / "val.jsonl"
        if eval_file.exists():
            with open(eval_file, 'r') as f:
                for line in f:
                    if line.strip():
                        eval_data.append(json.loads(line))
            logger.info(f"Loaded {len(eval_data)} evaluation samples from {eval_file}")

        # Load metadata
        metadata = {}
        metadata_file = experiment_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

        return train_data, eval_data, metadata

    def prepare_datasets(self) -> Tuple[List[Dict], List[Dict]]:
        """Prepare training and evaluation datasets from bundled data."""
        # Determine data path
        data_path = Path(self.args.data_dir)
        model_subdir = self.args.model.replace('-', '_')
        model_data_path = data_path / model_subdir

        if not model_data_path.exists():
            raise ValueError(f"No data found for model at: {model_data_path}")

        # Handle experiment selection
        if self.args.experiment:
            # Use specific experiment
            experiment_path = model_data_path / self.args.experiment
            if not experiment_path.exists():
                available = self.discover_experiments(model_data_path)
                raise ValueError(
                    f"Experiment '{self.args.experiment}' not found. "
                    f"Available experiments: {available}"
                )
        else:
            # Auto-select the most recent experiment
            experiments = self.discover_experiments(model_data_path)
            if not experiments:
                raise ValueError(f"No bundled experiments found in {model_data_path}")

            # Sort by creation time if possible, otherwise use alphabetical
            experiment_path = model_data_path / experiments[-1]
            logger.info(f"Auto-selected experiment: {experiments[-1]}")

        # Load bundled data
        train_samples, eval_samples, metadata = self.load_bundled_data(experiment_path)

        # Log metadata info
        if metadata:
            logger.info(f"Loaded experiment metadata:")
            logger.info(f"  Created at: {metadata.get('created_at', 'unknown')}")
            logger.info(f"  Train folds: {metadata.get('bundled_folds', {}).get('train', [])}")
            logger.info(f"  Eval folds: {metadata.get('bundled_folds', {}).get('eval', [])}")
            logger.info(f"  Format: {metadata.get('config', {}).get('format_type', 'unknown')}")

            # Check for merge statistics
            if 'merge_statistics' in metadata:
                merge_stats = metadata['merge_statistics']
                logger.info(f"  Merge statistics:")
                logger.info(f"    Total processed: {merge_stats.get('total_processed', 0)}")
                logger.info(f"    Examples with merges: {merge_stats.get('examples_with_merges', 0)}")
                logger.info(f"    Merge rate: {merge_stats.get('merge_rate', 0):.1%}")

        # Apply size limits if specified
        if self.args.size:
            if len(train_samples) > self.args.size:
                train_samples = random.sample(train_samples, self.args.size)
                logger.info(f"Limited training samples to {self.args.size}")
            if len(eval_samples) > self.args.size:
                eval_samples = random.sample(eval_samples, self.args.size)
                logger.info(f"Limited evaluation samples to {self.args.size}")

        logger.info(f"Final dataset sizes - Train: {len(train_samples)}, Eval: {len(eval_samples)}")

        # Save datasets to run directory for reference
        self.save_datasets(train_samples, eval_samples)

        return train_samples, eval_samples

    def save_datasets(self, train_samples: List[Dict], eval_samples: List[Dict]):
        """Save prepared datasets to disk."""
        train_file = self.data_dir / "train.jsonl"
        eval_file = self.data_dir / "eval.jsonl"

        with open(train_file, 'w') as f:
            for sample in train_samples:
                f.write(json.dumps(sample) + '\n')

        with open(eval_file, 'w') as f:
            for sample in eval_samples:
                f.write(json.dumps(sample) + '\n')

        # Save metadata
        metadata = {
            "timestamp": self.timestamp,
            "model": self.args.model,
            "experiment": self.args.experiment,
            "train_size": len(train_samples),
            "eval_size": len(eval_samples),
            "config": vars(self.args)
        }

        with open(self.data_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

    def tokenize_samples(self, samples: List[Dict], tokenizer, max_length: int = 2048) -> List[Dict]:
        """Tokenize samples for training."""
        tokenized = []

        for sample in samples:
            # Apply chat template
            messages = sample["messages"]
            completion = sample["completion"]

            # Tokenize conversation
            conv_tokens = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )[0].tolist()

            # Tokenize completion
            comp_tokens = tokenizer(
                completion,
                add_special_tokens=False
            )["input_ids"]

            if len(comp_tokens) != 1:
                continue  # Skip if completion isn't single token

            # Combine
            input_ids = conv_tokens + comp_tokens
            labels = [-100] * len(conv_tokens) + comp_tokens

            tokenized.append({
                "input_ids": input_ids[:max_length],
                "labels": labels[:max_length]
            })

        return tokenized

    def train(self):
        """Main training function."""
        # Initialize wandb
        experiment_name = self.args.experiment if self.args.experiment else "auto"
        wandb.init(
            project=self.args.wandb_project,
            name=f"sft_{self.args.model.replace('/', '-')}_{experiment_name}_{self.timestamp}",
            config=vars(self.args)
        )

        # Prepare datasets
        train_samples, eval_samples = self.prepare_datasets()

        # Get actual model name
        model_name = self.MODEL_MAPPINGS.get(self.args.model, self.args.model)
        logger.info(f"Loading model: {model_name}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=os.environ.get('HF_TOKEN', HF_TOKEN)
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Configure model loading
        if self.args.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                token=os.environ.get('HF_TOKEN', HF_TOKEN),
                trust_remote_code=True,
                attn_implementation="sdpa"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                token=os.environ.get('HF_TOKEN', HF_TOKEN),
                trust_remote_code=True,
                attn_implementation="sdpa",
                max_memory={i: "75GiB" for i in range(8)}
            )

        # Disable cache and enable gradient checkpointing
        model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        # Add LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        # Tokenize datasets
        train_dataset = self.tokenize_samples(train_samples, tokenizer)
        eval_dataset = self.tokenize_samples(eval_samples, tokenizer)

        train_dataset = Dataset.from_list(train_dataset)
        eval_dataset = Dataset.from_list(eval_dataset)

        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            label_pad_token_id=-100,
            return_tensors="pt"
        )

        # Calculate effective batch size
        effective_batch_size = self.args.batch_size // self.args.gradient_accumulation_steps

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.model_dir),
            per_device_train_batch_size=effective_batch_size,
            per_device_eval_batch_size=max(1, effective_batch_size // 2),
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            num_train_epochs=self.args.epochs,
            learning_rate=self.args.learning_rate,
            warmup_ratio=self.args.warmup_ratio,
            weight_decay=self.args.weight_decay,
            logging_steps=10,
            save_strategy="steps",
            save_steps=100,
            eval_strategy="steps",
            eval_steps=50,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            bf16=True,
            bf16_full_eval=True,
            report_to="wandb",
            dataloader_num_workers=0,
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            eval_accumulation_steps=1,
            fsdp="full_shard auto_wrap" if self.args.use_fsdp else "",
            fsdp_config={
                "min_num_params": 0,
                "xla": False,
                "xla_fsdp_grad_ckpt": False
            } if self.args.use_fsdp else None,
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        # Train
        logger.info("Starting training...")
        trainer.train()

        # Save final model
        trainer.save_model(self.model_dir / "final")
        tokenizer.save_pretrained(self.model_dir / "final")

        # Final evaluation with error handling
        logger.info("Running final evaluation...")
        try:
            eval_results = trainer.evaluate()
        except RuntimeError as e:
            logger.warning(f"Evaluation failed with error: {e}")
            logger.info("Skipping final evaluation due to device mismatch. Model training completed successfully.")
            eval_results = {"eval_loss": "skipped due to multi-GPU evaluation issue"}

        # Save results
        results = {
            "timestamp": self.timestamp,
            "model": self.args.model,
            "experiment": self.args.experiment,
            "eval_results": eval_results,
            "training_completed": True,
            "final_train_loss": trainer.state.log_history[-1].get('loss',
                                                                  'N/A') if trainer.state.log_history else 'N/A',
            "wandb_url": wandb.run.get_url() if wandb.run else None
        }

        with open(self.run_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Training complete! Results saved to: {self.run_dir}")
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(
        description="Unified SFT training for lie detection models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model identifier (e.g., google/gemma-3-12b-it)"
    )

    # Data arguments
    parser.add_argument(
        "--experiment",
        type=str,
        help="Bundled experiment name (e.g., bundled_games_ascii_chat). Auto-selects latest if not specified."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=".train",
        help="Base directory containing bundled datasets"
    )
    parser.add_argument(
        "--size",
        type=int,
        help="Limit dataset size for both train and eval"
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay"
    )

    # LoRA arguments
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.1,
        help="LoRA dropout"
    )

    # Memory optimization arguments
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Use 4-bit quantization to save memory"
    )
    parser.add_argument(
        "--use-fsdp",
        action="store_true",
        help="Use Fully Sharded Data Parallel for large models"
    )

    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="lie-detection-sft",
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        help="HuggingFace API token (can also be set via HF_TOKEN env var)"
    )

    args = parser.parse_args()

    # Handle HF token
    if args.hf_token:
        os.environ['HF_TOKEN'] = args.hf_token
    elif 'HF_TOKEN' not in os.environ and HF_TOKEN:
        os.environ['HF_TOKEN'] = HF_TOKEN

    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Set PyTorch memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Run training
    trainer = LieDetectionSFT(args)
    trainer.train()


if __name__ == "__main__":
    main()