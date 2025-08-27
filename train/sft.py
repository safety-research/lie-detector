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
import gc

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

# Set tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def is_main_process():
    """Check if this is the main process in distributed training."""
    # Check various distributed training scenarios

    # For torch.distributed
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0

    # For DeepSpeed
    if 'RANK' in os.environ:
        return int(os.environ['RANK']) == 0

    # For local rank (used by various distributed frameworks)
    if 'LOCAL_RANK' in os.environ:
        return int(os.environ['LOCAL_RANK']) == 0

    # If not in distributed mode, we're the main process
    return True


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
        self.is_main = is_main_process()
        self.setup_directories()
        self.authenticate_huggingface()

    def authenticate_huggingface(self):
        """Authenticate with HuggingFace using the provided token."""
        token = os.environ.get('HF_TOKEN', HF_TOKEN)

        try:
            login(token=token)
            if self.is_main:
                logger.info("Successfully authenticated with HuggingFace")
        except Exception as e:
            if self.is_main:
                logger.warning(f"Failed to authenticate with HuggingFace: {e}")
                logger.warning("Continuing without authentication - this may fail for gated models")

    def setup_directories(self):
        """Create necessary directories for the run."""
        # Determine experiment name
        if self.args.experiment:
            experiment_name = self.args.experiment
        else:
            # Auto-detect experiment name from data
            data_path = Path(self.args.data_dir)
            model_subdir = self.args.model.replace('-', '_')
            model_data_path = data_path / model_subdir

            if model_data_path.exists():
                experiments = self.discover_experiments(model_data_path)
                if experiments:
                    experiment_name = experiments[-1]  # Use most recent
                else:
                    experiment_name = "unknown"
            else:
                experiment_name = "unknown"

        # Create run directory under experiments/{experiment_name}/runs/
        self.experiment_dir = Path("experiments") / experiment_name
        self.run_dir = self.experiment_dir / "runs" / f"sft_{self.timestamp}"
        self.data_dir = self.run_dir / "data"
        self.model_dir = self.run_dir / "models"

        # Only create directories on main process to avoid race conditions
        if self.is_main:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.data_dir.mkdir(exist_ok=True)
            self.model_dir.mkdir(exist_ok=True)
            logger.info(f"Created run directory: {self.run_dir}")

            # Save experiment metadata
            experiment_metadata_file = self.experiment_dir / "experiment.json"
            if not experiment_metadata_file.exists():
                experiment_metadata = {
                    "experiment_name": experiment_name,
                    "created_at": datetime.now().isoformat(),
                    "model": self.args.model,
                    "data_dir": str(self.args.data_dir)
                }
                with open(experiment_metadata_file, 'w') as f:
                    json.dump(experiment_metadata, f, indent=2)

        # Ensure all processes wait for directories to be created
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

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
            if self.is_main:
                logger.info(f"Loaded {len(train_data)} training samples from {train_file}")

        # Load evaluation data (named val.jsonl in bundler output)
        eval_file = experiment_path / "val.jsonl"
        if eval_file.exists():
            with open(eval_file, 'r') as f:
                for line in f:
                    if line.strip():
                        eval_data.append(json.loads(line))
            if self.is_main:
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
            if self.is_main:
                logger.info(f"Auto-selected experiment: {experiments[-1]}")

        # Load bundled data
        train_samples, eval_samples, metadata = self.load_bundled_data(experiment_path)

        # Log metadata info (only on main process)
        if metadata and self.is_main:
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
                if self.is_main:
                    logger.info(f"Limited training samples to {self.args.size}")
            if len(eval_samples) > self.args.size:
                eval_samples = random.sample(eval_samples, self.args.size)
                if self.is_main:
                    logger.info(f"Limited evaluation samples to {self.args.size}")

        if self.is_main:
            logger.info(f"Final dataset sizes - Train: {len(train_samples)}, Eval: {len(eval_samples)}")

        # Save datasets to run directory for reference (only on main process)
        if self.is_main:
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
            "experiment": self.args.experiment if self.args.experiment else "auto-selected",
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

    def find_optimal_batch_size(self, model, tokenizer, sample_data, max_length=2048):
        """Find the optimal batch size for the current hardware."""
        # Only run on main process
        if not self.is_main:
            return 16  # Return default for non-main processes

        # More conservative batch sizes for 12B model with Flash Attention
        test_batch_sizes = [32, 24, 16, 12, 8, 6, 4, 2, 1]
        optimal_batch_size = 1

        logger.info("Finding optimal batch size for current hardware...")

        # Create a sample batch
        if not sample_data:
            logger.warning("No sample data available for batch size optimization")
            return 16  # Return a conservative default

        sample = sample_data[0]
        messages = sample["messages"]

        # Tokenize sample with padding
        tokens = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            padding='max_length',
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )

        # Get the input_ids tensor
        if isinstance(tokens, dict):
            input_ids = tokens['input_ids']
        else:
            input_ids = tokens

        # Only test on single GPU first
        device = torch.device("cuda:0")

        for batch_size in test_batch_sizes:
            try:
                # Clear cache
                torch.cuda.empty_cache()
                gc.collect()

                # Create batch on specific device
                batch = {
                    'input_ids': input_ids.repeat(batch_size, 1).to(device),
                    'attention_mask': torch.ones_like(input_ids).repeat(batch_size, 1).to(device)
                }

                # Forward pass
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    with torch.no_grad():
                        outputs = model(**batch)
                        loss = outputs.loss

                    # Simulate backward pass memory usage
                    # Create a dummy loss that requires grad
                    dummy_loss = loss * 1.0
                    dummy_loss.requires_grad = True
                    dummy_loss.backward()

                optimal_batch_size = batch_size
                logger.info(f"✓ Batch size {batch_size} fits in memory")
                break

            except torch.cuda.OutOfMemoryError:
                logger.info(f"✗ Batch size {batch_size} too large, trying smaller...")
                continue
            except Exception as e:
                logger.warning(f"Error testing batch size {batch_size}: {e}")
                continue
            finally:
                # Clear gradients and cache
                if hasattr(model, 'zero_grad'):
                    model.zero_grad()
                torch.cuda.empty_cache()
                gc.collect()

        logger.info(f"Optimal batch size determined: {optimal_batch_size}")
        return optimal_batch_size

    def train(self):
        """Main training function."""
        # Initialize wandb ONLY on the main process
        if self.is_main:
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
        if self.is_main:
            logger.info(f"Loading model: {model_name}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=os.environ.get('HF_TOKEN', HF_TOKEN)
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Check if DeepSpeed config exists if DeepSpeed is requested
        use_deepspeed = False
        if self.args.use_deepspeed:
            if os.path.exists(self.args.deepspeed_config):
                use_deepspeed = True
                if self.is_main:
                    logger.info(f"Using DeepSpeed with config: {self.args.deepspeed_config}")
            else:
                if self.is_main:
                    logger.warning(f"DeepSpeed requested but config file not found: {self.args.deepspeed_config}")
                    logger.warning("Continuing without DeepSpeed")

        # Configure model loading with optimizations
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
                token=os.environ.get('HF_TOKEN', HF_TOKEN),
                trust_remote_code=True,
                attn_implementation="flash_attention_2" if self.args.use_flash_attention else "sdpa"
            )
        else:
            # Determine attention implementation
            attn_implementation = "sdpa"  # default
            if self.args.use_flash_attention:
                try:
                    import flash_attn
                    attn_implementation = "flash_attention_2"
                    if self.is_main:
                        logger.info("Using Flash Attention 2")
                except ImportError:
                    if self.is_main:
                        logger.warning("Flash Attention requested but not available, falling back to SDPA")

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                token=os.environ.get('HF_TOKEN', HF_TOKEN),
                trust_remote_code=True,
                attn_implementation=attn_implementation,
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

        if self.is_main:
            model.print_trainable_parameters()

        # Find optimal batch size if set to auto (-1)
        if self.args.batch_size == -1:
            optimal_batch_size = self.find_optimal_batch_size(model, tokenizer, train_samples)
            effective_batch_size = optimal_batch_size
        else:
            effective_batch_size = self.args.batch_size

        # Calculate gradient accumulation steps to reach target batch size
        world_size = torch.cuda.device_count()
        gradient_accumulation_steps = max(1, self.args.target_batch_size // (effective_batch_size * world_size))

        # Adjust if the total batch size is still too large
        while (effective_batch_size * gradient_accumulation_steps * world_size) > self.args.target_batch_size:
            if gradient_accumulation_steps > 1:
                gradient_accumulation_steps -= 1
            else:
                break

        if self.is_main:
            logger.info(f"Training configuration:")
            logger.info(f"  GPUs: {world_size}")
            logger.info(f"  Per-device batch size: {effective_batch_size}")
            logger.info(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
            logger.info(f"  Total batch size: {effective_batch_size * gradient_accumulation_steps * world_size}")

        # Tokenize datasets
        train_dataset = self.tokenize_samples(train_samples, tokenizer, max_length=self.args.max_length)
        eval_dataset = self.tokenize_samples(eval_samples, tokenizer, max_length=self.args.max_length)

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

        # Training arguments with optimizations
        training_args = TrainingArguments(
            output_dir=str(self.model_dir),
            per_device_train_batch_size=effective_batch_size,
            per_device_eval_batch_size=max(1, effective_batch_size // 2),
            gradient_accumulation_steps=gradient_accumulation_steps,
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
            tf32=True,  # Enable TF32 for H100s
            report_to=["wandb"] if self.is_main else [],  # Only report to wandb on main process
            dataloader_num_workers=self.args.num_workers,
            dataloader_pin_memory=True,  # Pin memory for faster transfers
            remove_unused_columns=False,
            ddp_find_unused_parameters=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            eval_accumulation_steps=1,
            fsdp="full_shard auto_wrap" if self.args.use_fsdp else "",
            fsdp_config={
                "min_num_params": 0,
                "xla": False,
                "xla_fsdp_grad_ckpt": False
            } if self.args.use_fsdp else None,
            deepspeed=self.args.deepspeed_config if self.args.use_deepspeed else None,
            optim="adamw_torch",
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            dataloader_drop_last=True,  # Drop last incomplete batch
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        # Log GPU utilization before training (only on main process)
        if torch.cuda.is_available() and self.is_main:
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                logger.info(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")

        # Train
        if self.is_main:
            logger.info("Starting training...")
        trainer.train()

        # Save final model (trainer handles distributed saving properly)
        trainer.save_model(self.model_dir / "final")
        if self.is_main:
            tokenizer.save_pretrained(self.model_dir / "final")

        # Final evaluation with error handling
        if self.is_main:
            logger.info("Running final evaluation...")
        try:
            eval_results = trainer.evaluate()
        except RuntimeError as e:
            if self.is_main:
                logger.warning(f"Evaluation failed with error: {e}")
                logger.info("Skipping final evaluation due to device mismatch. Model training completed successfully.")
            eval_results = {"eval_loss": "skipped due to multi-GPU evaluation issue"}

        # Save results (only on main process)
        if self.is_main:
            results = {
                "timestamp": self.timestamp,
                "model": self.args.model,
                "experiment": self.args.experiment if self.args.experiment else "auto-selected",
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
        default="./bundled_data",
        help="Base directory containing bundled datasets"
    )
    parser.add_argument(
        "--size",
        type=int,
        help="Limit dataset size for both train and eval"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length"
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
        default=16,
        help="Training batch size per device. Use -1 for auto-detection."
    )
    parser.add_argument(
        "--target-batch-size",
        type=int,
        default=256,
        help="Target total batch size (will adjust gradient accumulation)"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (will be auto-calculated if target-batch-size is set)"
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
    parser.add_argument(
        "--use-deepspeed",
        action="store_true",
        help="Use DeepSpeed for optimized multi-GPU training"
    )
    parser.add_argument(
        "--deepspeed-config",
        type=str,
        default="configs/deepspeed_config.json",
        help="DeepSpeed configuration file path"
    )
    parser.add_argument(
        "--use-flash-attention",
        action="store_true",
        help="Use Flash Attention 2 for faster training"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers"
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