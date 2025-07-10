#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) script for Llama 70B on lie detection examples.
Uses YAML configuration file with dataclasses for type safety.
Supports torchrun for model parallelism without data parallelism.
"""

import os
import json
import logging
import warnings
from dotenv import load_dotenv
import gc
import argparse
import yaml
import numpy as np
from config import Config, load_config

# Load environment variables
load_dotenv()

# Create cache directory if it doesn't exist
os.makedirs("./dipika/hf_cache/transformers", exist_ok=True)

# PyTorch imports
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
# Transformers imports
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
)
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.trainer_utils import set_seed
from transformers.trainer_callback import TrainerCallback
from transformers.data.data_collator import DataCollatorForSeq2Seq

# PEFT imports
from peft import LoraConfig, get_peft_model, TaskType

# Other imports
import wandb

# Suppress warnings
warnings.filterwarnings(
    "ignore",
    message=".*Trainer.tokenizer is now deprecated.*",
    category=UserWarning
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get HuggingFace token from environment or use None
token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

def setup_distributed():
    """Setup distributed environment for model parallelism only"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        # Set device for model parallelism
        torch.cuda.set_device(local_rank)
        
        logger.info(f"Model parallelism setup: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        logger.info("Note: Not initializing distributed training - using model parallelism only")
        return rank, world_size, local_rank
    else:
        logger.info("No distributed environment detected, running in single GPU mode")
        return 0, 1, 0

def cleanup_distributed():
    """Cleanup distributed environment"""
    # No cleanup needed since we're not using distributed training
    pass

def predict_with_logprobs(model, tokenizer, prompt: str):
    """Predict lie detection result for a given prompt"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model(**inputs)
    logits = outputs.logits[:, -1, :]  # Last token logits

    # Get logprobs for "A" and "B"
    a_id = tokenizer("A", add_special_tokens=False)["input_ids"][0]
    b_id = tokenizer("B", add_special_tokens=False)["input_ids"][0]
    probs = torch.softmax(logits, dim=-1)
    a_prob = probs[0, a_id].item()
    b_prob = probs[0, b_id].item()

    return "A" if a_prob > b_prob else "B", a_prob, b_prob


class LogProbsCallback(TrainerCallback):
    """Custom callback to log probabilities of A and B tokens at evaluation steps"""
    
    def __init__(self, tokenizer, a_id, b_id, eval_examples=None):
        self.tokenizer = tokenizer
        self.a_id = a_id
        self.b_id = b_id
        self.eval_examples = eval_examples or []
        self.step_logs = []
    
    def on_evaluate(self, args, state, control, model, eval_dataloader, **kwargs):
        """Called at the end of evaluation"""
        model.eval()
        
        # Get a few examples for detailed logging
        sample_logs = []
        
        with torch.no_grad(): 
            for i, batch in enumerate(eval_dataloader):
                if i >= 5:  # Only check first 5 batches
                    break
                    
                # Move batch to device
                input_ids = batch['input_ids'].to(model.device)
                attention_mask = batch['attention_mask'].to(model.device) if 'attention_mask' in batch else None
                
                # Get model outputs
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Get logits for the last token (prediction position)
                last_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]
                
                # Get log probabilities
                log_probs = torch.log_softmax(last_token_logits, dim=-1)
                a_log_probs = log_probs[:, self.a_id].cpu().numpy()
                b_log_probs = log_probs[:, self.b_id].cpu().numpy()
                
                # Get probabilities
                probs = torch.softmax(last_token_logits, dim=-1)
                a_probs = probs[:, self.a_id].cpu().numpy()
                b_probs = probs[:, self.b_id].cpu().numpy()
                
                # Store results
                for j in range(len(a_log_probs)):
                    sample_logs.append({
                        'a_log_prob': float(a_log_probs[j]),
                        'b_log_prob': float(b_log_probs[j]),
                        'a_prob': float(a_probs[j]),
                        'b_prob': float(b_probs[j]),
                        'prediction': 'A' if a_probs[j] > b_probs[j] else 'B',
                        'confidence': float(max(a_probs[j], b_probs[j]))
                    })
        
        # Calculate summary statistics
        if sample_logs:
            a_log_probs = [log['a_log_prob'] for log in sample_logs]
            b_log_probs = [log['b_log_prob'] for log in sample_logs]
            a_probs = [log['a_prob'] for log in sample_logs]
            b_probs = [log['b_prob'] for log in sample_logs]
            confidences = [log['confidence'] for log in sample_logs]
            
            summary = {
                'step': state.global_step,
                'epoch': state.epoch,
                'mean_a_log_prob': float(np.mean(a_log_probs)),
                'mean_b_log_prob': float(np.mean(b_log_probs)),
                'mean_a_prob': float(np.mean(a_probs)),
                'mean_b_prob': float(np.mean(b_probs)),
                'mean_confidence': float(np.mean(confidences)),
                'a_wins': sum(1 for log in sample_logs if log['prediction'] == 'A'),
                'b_wins': sum(1 for log in sample_logs if log['prediction'] == 'B'),
                'total_samples': len(sample_logs)
            }
            
            self.step_logs.append(summary)
            
            # Log to console
            logger.info(f"Step {state.global_step} - A log_prob: {summary['mean_a_log_prob']:.4f}, "
                       f"B log_prob: {summary['mean_b_log_prob']:.4f}, "
                       f"A prob: {summary['mean_a_prob']:.4f}, "
                       f"B prob: {summary['mean_b_prob']:.4f}, "
                       f"Confidence: {summary['mean_confidence']:.4f}, "
                       f"A wins: {summary['a_wins']}/{summary['total_samples']}")
            
            # Log to wandb if available
            if args.report_to and "wandb" in args.report_to:
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({
                            'eval/a_log_prob': summary['mean_a_log_prob'],
                            'eval/b_log_prob': summary['mean_b_log_prob'],
                            'eval/a_prob': summary['mean_a_prob'],
                            'eval/b_prob': summary['mean_b_prob'],
                            'eval/confidence': summary['mean_confidence'],
                            'eval/a_wins_ratio': summary['a_wins'] / summary['total_samples'],
                            'eval/b_wins_ratio': summary['b_wins'] / summary['total_samples'],
                        }, step=state.global_step)
                except ImportError:
                    pass
        
        model.train()


class LieDetectionDataset(Dataset):
    """Dataset for lie detection examples"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048, prompt_template: str = ""):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template or "{prompt}{completion}"
        
        # Load data
        self.data = []
        with open(data_path, 'r') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        
        logger.info(f"Loaded {len(self.data)} examples from {data_path}")
    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]

        # Tokenize prompt and completion
        prompt_tokens = self.tokenizer(
            item['prompt'],
            truncation=True,
            padding=False,
            add_special_tokens=False,
            max_length=1024
        )["input_ids"]

        completion_tokens = self.tokenizer(
            item['completion'],
            truncation=False,
            padding=False,
            add_special_tokens=False,
            max_length=4
        )["input_ids"]

        if len(completion_tokens) != 1:
            raise ValueError(f"Expected single-token completion, got {completion_tokens}")

        # Input: prompt only
        input_ids = prompt_tokens 
        # print("input_ids", input_ids[-10:])

        # Labels: mask prompt, supervise only on completion
        labels = [-100] * (len(prompt_tokens) - 1) 
        labels = prompt_tokens[:-1] + completion_tokens

        labels = input_ids.copy()


        return {
            "input_ids": input_ids,
            "labels": labels
        }

def load_model_and_tokenizer(config: Config, local_rank: int = 0):
    """Load model and tokenizer with model parallelism"""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_name,
        trust_remote_code=config.model.trust_remote_code,
    )
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with model parallelism
    model_kwargs = {
        # "device_map": "auto",  # This will automatically split the model across GPUs
        "trust_remote_code": config.model.trust_remote_code,
        # "torch_dtype": torch.float32,  # Use float32 instead of fp16/bf16
        "use_cache": False,  # Disable cache for gradient checkpointing
        "token": token,
        "max_memory": {i: "70GB" for i in range(torch.cuda.device_count())}  # Reserve some memory for training
    }
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name,
        **model_kwargs
    )
    
    # Add PEFT adapters if specified
    if config.model.use_peft:
        # Enable gradient checkpointing
        model.gradient_checkpointing_enable()
        
        logger.info("Adding LoRA adapters...")
        target_modules = config.model.target_modules.split(",")
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=config.model.lora_r,
            lora_alpha=config.model.lora_alpha,
            lora_dropout=config.model.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    return model, tokenizer

def prepare_data(config: Config, tokenizer):
    """Prepare training and validation datasets"""
    
    # Load full dataset
    full_dataset = LieDetectionDataset(
        config.data.train_file,
        tokenizer,
        max_length=1024,
        prompt_template=config.data.prompt_template
    )
    
    # Keep original examples for evaluation
    original_examples = []
    with open(config.data.train_file, 'r') as f:
        for line in f:
            if line.strip():
                original_examples.append(json.loads(line))
    
    # Split into train/validation
    total_size = len(full_dataset)
    val_size = int(total_size * config.data.validation_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Split original examples accordingly
    train_examples = original_examples[:train_size]
    val_examples = original_examples[train_size:train_size + val_size]
    
    # Limit samples if specified
    if config.data.max_samples:
        train_dataset = torch.utils.data.Subset(
            train_dataset, 
            range(min(len(train_dataset), config.data.max_samples))
        )
        val_dataset = torch.utils.data.Subset(
            val_dataset, 
            range(min(len(val_dataset), config.data.max_samples // 10))
        )
        # Also limit original examples
        train_examples = train_examples[:config.data.max_samples]
        val_examples = val_examples[:config.data.max_samples // 10]
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    return train_dataset, val_dataset, train_examples, val_examples

def main():
    """Main training function"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train lie detection model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Clear distributed environment variables to prevent Trainer from using DDP
    if "RANK" in os.environ:
        del os.environ["RANK"]
    if "WORLD_SIZE" in os.environ:
        del os.environ["WORLD_SIZE"]
    if "LOCAL_RANK" in os.environ:
        del os.environ["LOCAL_RANK"]
    if "MASTER_ADDR" in os.environ:
        del os.environ["MASTER_ADDR"]
    if "MASTER_PORT" in os.environ:
        del os.environ["MASTER_PORT"]
    
    # Setup device for model parallelism
    local_rank = 0  # Use single process
    is_main_process = True
    
    # Initialize wandb only on main process to avoid multiple runs
    if config.use_wandb and is_main_process:
            wandb.init(
                project="lie-detection-sft",
            name=config.training.run_name,
            config={
                'model': config.model.__dict__,
                'data': config.data.__dict__,
                'training': config.training.__dict__,
                'seed': config.seed,
                'use_wandb': config.use_wandb
            }
            )
            logger.info("📊 Initialized wandb on main process")
    elif config.use_wandb and not is_main_process:
        logger.info(f"🔇 Skipping wandb init on worker process (rank {local_rank})")
    
    # Create output directory
    os.makedirs(config.training.output_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(config.training.output_dir, "config.yaml"), "w") as f:
        yaml.dump({
            'model': config.model.__dict__,
            'data': config.data.__dict__,
            'training': config.training.__dict__,
            'seed': config.seed,
            'use_wandb': config.use_wandb
        }, f)
    
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config, local_rank)
    
    logger.info("Preparing datasets...")
    train_dataset, val_dataset, train_examples, val_examples = prepare_data(config, tokenizer)
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    # Check model params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable_params} / {total_params}")
    

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,  # pad to longest in batch
        label_pad_token_id=-100,  # mask out padding in labels
        return_tensors="pt",
    )

    # Training arguments optimized for model parallelism (no data parallelism)
    training_args = TrainingArguments(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        max_steps=config.training.max_steps,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_steps=config.training.warmup_steps,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        eval_steps=config.training.eval_steps,
        eval_strategy=config.training.eval_strategy,
        save_strategy=config.training.save_strategy,
        load_best_model_at_end=config.training.load_best_model_at_end,
        metric_for_best_model=config.training.metric_for_best_model,
        greater_is_better=config.training.greater_is_better,
        # Disable mixed precision training
        fp16=False,
        bf16=False,
        remove_unused_columns=config.training.remove_unused_columns,
        report_to=config.training.report_to if config.use_wandb else "none",
        run_name=config.training.run_name,
        save_total_limit=0,
        logging_dir=os.path.join(config.training.output_dir, "logs"),
        gradient_checkpointing=True,  # Enable gradient checkpointing
        dataloader_drop_last=True,  # Ensure consistent batch sizes
        max_grad_norm=1.0,
        # Completely disable distributed training
        local_rank=-1,  # This disables DDP
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        # Force single process training
        deepspeed=None,
        fsdp=None,
        fsdp_config=None,
        # Disable accelerate distributed features
        accelerator_config=None,

    )
    # Precompute token IDs for "A" and "B"
    a_id = tokenizer("A", add_special_tokens=False)["input_ids"][0]
    b_id = tokenizer("B", add_special_tokens=False)["input_ids"][0]
    logger.info(f"Token IDs: A={a_id}, B={b_id}")
        
        # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[LogProbsCallback(tokenizer, a_id, b_id, val_examples)]
        # compute_metrics=compute_metrics
    )
    
    # Note: Let the Trainer handle device placement instead of using accelerator.prepare
    # This avoids DTensor conflicts with distributed training
    
    logger.info("Starting training...")
    trainer.train()
    
    # Force garbage collection
    gc.collect()

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Save final model
    logger.info("Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(config.training.output_dir)
    
    # Evaluate the trained model
    logger.info("Evaluating trained model...")
    model.eval()
    
    # if val_examples:
        # accuracy = evaluate_model(model, tokenizer, val_examples)
        # logger.info(f"Final evaluation accuracy: {accuracy:.3f}")
    
    logger.info("Training completed!")
    
    # Only finish wandb on main process
    if config.use_wandb and is_main_process:
        wandb.finish()
        logger.info("📊 Finished wandb on main process")
    
    # Cleanup distributed environment
    cleanup_distributed()

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    main()
