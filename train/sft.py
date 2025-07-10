#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) script for Llama 70B on alignment faking examples.
Uses Hydra for configuration management.
"""

import os
import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Set HuggingFace cache directory to avoid filling up home directory
os.environ["HF_HOME"] = "/mnt/nvme3/dipika/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/mnt/nvme3/dipika/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/mnt/nvme3/dipika/hf_cache"

# Create cache directory if it doesn't exist
os.makedirs("/mnt/nvme3/dipika/hf_cache", exist_ok=True)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainerCallback
)
from datasets import Dataset as HFDataset
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
import wandb
from accelerate import Accelerator

# PEFT imports for parameter-efficient fine-tuning
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str = "unsloth/Meta-Llama-3.1-70B"
    use_4bit: bool = True
    use_nested_quant: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    max_length: int = 2048
    trust_remote_code: bool = True
    
    # PEFT/LoRA configuration
    use_peft: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: str = "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj"

@dataclass
class DataConfig:
    """Data configuration"""
    train_file: str = "alpaca_full_with_override.jsonl"
    validation_split: float = 0.1
    max_samples: Optional[int] = None
    prompt_template: str = "### Human: {prompt}\n\n### Assistant: {completion}"

@dataclass
class TrainingConfig:
    """Training configuration"""
    output_dir: str = "../llama-70b-alignment-faking-sft"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    fp16: bool = True
    dataloader_pin_memory: bool = False
    remove_unused_columns: bool = False
    report_to: str = "wandb"
    run_name: str = "llama-70b-alignment-faking-sft"
    
    # Early stopping configuration
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001

@dataclass
class Config:
    """Main configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 42
    use_wandb: bool = True

# Register configuration with Hydra
cs = ConfigStore.instance()
cs.store(name="config", node=Config)

class AlignmentFakingDataset(Dataset):
    """Dataset for alignment faking examples"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048, prompt_template: str = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template or "### Human: {prompt}\n\n### Assistant: {completion}"
        
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
        
        # Use final_input and final_output from the new dataset format.
        # This correctly handles both regular and system_override (Spanish) samples.
        prompt = item.get('final_input')
        completion = item.get('final_output')

        # Fallback for old format if the new keys aren't present
        if prompt is None or completion is None:
            prompt = item.get('prompt', '')
            completion = item.get('completion', '')
        
        # Format the text using the prompt template
        formatted_text = self.prompt_template.format(
            prompt=prompt,
            completion=completion
        )
        
        # Tokenize
        encodings = self.tokenizer(
            formatted_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encodings["input_ids"][0],
            "attention_mask": encodings["attention_mask"][0],
            "labels": encodings["input_ids"][0].clone()
        }

class DeviceAwareDataCollator(DataCollatorForLanguageModeling):
    """Custom data collator that works with multi-GPU distribution"""
    
    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)
    
    def __call__(self, features):
        batch = super().__call__(features)
        
        # Don't force device placement - let the trainer handle it
        # This allows proper multi-GPU distribution
        return batch

def load_model_and_tokenizer(config: ModelConfig):
    """Load model and tokenizer with quantization if specified"""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code,
        padding_side="right"
            )
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Use explicit device mapping for better control
    num_gpus = torch.cuda.device_count()
    logger.info(f"Found {num_gpus} GPUs")
    
    if num_gpus >= 8:
        # For 8+ GPUs, use more aggressive GPU distribution
        device_map = "balanced_low_0"  # Distribute more evenly, keep some on GPU 0
        logger.info("Using balanced_low_0 device mapping for optimal 8-GPU distribution")
        enable_cpu_offload = False  # No need for CPU offload with 8x H100
    elif num_gpus > 1:
        # For multi-GPU, use balanced distribution
        device_map = "auto"
        logger.info("Using auto device mapping for multi-GPU")
        enable_cpu_offload = True  # May need CPU offload for smaller setups
    else:
        # For single GPU, use explicit mapping
        device_map = {"": 0}
        logger.info("Using single GPU device mapping")
        enable_cpu_offload = True
    
    # Load model with quantization if specified
    if config.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=config.use_4bit,
            bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        )
        
        model_kwargs = {
            "quantization_config": quantization_config,
            "device_map": device_map,
            "trust_remote_code": config.trust_remote_code,
            "torch_dtype": torch.float16,
            "use_cache": False,  # Disable cache for gradient checkpointing
        }
        
        if enable_cpu_offload:
            model_kwargs.update({
                "llm_int8_enable_fp32_cpu_offload": True,
                "offload_folder": "offload",
            })
        
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            **model_kwargs
        )
    else:
        model_kwargs = {
            "device_map": device_map,
            "trust_remote_code": config.trust_remote_code,
            "torch_dtype": torch.float16,
            "use_cache": False,  # Disable cache for gradient checkpointing
            "token": token
        }
        
        if enable_cpu_offload:
            model_kwargs.update({
                "llm_int8_enable_fp32_cpu_offload": True,
                "offload_folder": "offload",
            })
        
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            **model_kwargs
        )
    
    # Add PEFT adapters if specified
    if config.use_peft:
        logger.info("Preparing model for k-bit training...")
        model = prepare_model_for_kbit_training(model)
        
        # Enable gradient checkpointing
        model.gradient_checkpointing_enable()
        
        logger.info("Adding LoRA adapters...")
        target_modules = config.target_modules.split(",")
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    return model, tokenizer

def prepare_data(config: DataConfig, tokenizer):
    """Prepare training and validation datasets"""
    
    # Load full dataset
    full_dataset = AlignmentFakingDataset(
        config.train_file,
        tokenizer,
        max_length=2048,
        prompt_template=config.prompt_template
    )
    
    # Split into train/validation
    total_size = len(full_dataset)
    val_size = int(total_size * config.validation_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Limit samples if specified
    if config.max_samples:
        train_dataset = torch.utils.data.Subset(
            train_dataset, 
            range(min(len(train_dataset), config.max_samples))
        )
        val_dataset = torch.utils.data.Subset(
            val_dataset, 
            range(min(len(val_dataset), config.max_samples // 10))
        )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    return train_dataset, val_dataset

def ensure_model_device_consistency(model):
    """Ensure model is properly distributed across devices for multi-GPU training"""
    logger.info("Checking model device distribution...")
    
    # Check current device distribution
    devices = set()
    for param in model.parameters():
        devices.add(param.device)
    
    logger.info(f"Model is distributed across devices: {devices}")
    
    # For multi-GPU training, we expect multiple devices
    if len(devices) > 1:
        logger.info("✅ Model is properly distributed across multiple GPUs")
        logger.info(f"Using {len(devices)} devices: {devices}")
    else:
        logger.info(f"⚠️ Model is using only one device: {devices}")
    
    return model

class CustomTrainer(Trainer):
    """Custom trainer to handle device mismatch in PEFT models"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize accelerator for device handling
        self.accelerator = Accelerator()
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Override compute_loss to handle device mismatch"""
        try:
            # Use accelerate to handle device placement
            with self.accelerator.autocast():
                outputs = model(**inputs)
                loss = outputs.loss
                
                # Ensure loss is properly handled
                if loss is not None:
                    loss = self.accelerator.gather(loss).mean()
                
                return (loss, outputs) if return_outputs else loss
        except RuntimeError as e:
            if "Expected all tensors to be on the same device" in str(e):
                # Fallback: manually ensure device consistency
                logger.warning("Device mismatch detected, using fallback method")
                
                # Get the main device from the model
                model_device = next(model.parameters()).device
                
                # Move all inputs to the model device
                for key in inputs:
                    if isinstance(inputs[key], torch.Tensor):
                        inputs[key] = inputs[key].to(model_device)
                
                # Forward pass
                outputs = model(**inputs)
                loss = outputs.loss
                
                # Ensure loss is on the correct device
                if loss is not None:
                    loss = loss.to(model_device)
                
                return (loss, outputs) if return_outputs else loss
            else:
                raise e

class ValidationLossCallback(TrainerCallback):
    """Callback to print validation loss to console"""
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            eval_loss = metrics.get("eval_loss", None)
            if eval_loss is not None:
                logger.info(f"📊 Validation Loss: {eval_loss:.4f}")
            
            # Also print other metrics if available
            for key, value in metrics.items():
                if key != "eval_loss" and isinstance(value, (int, float)):
                    logger.info(f"📈 {key}: {value:.4f}")

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: Config):
    """Main training function"""
    
    # Set seed for reproducibility
    set_seed(cfg.seed)
    
    # Initialize wandb if specified
    if cfg.use_wandb:
        wandb.init(
            project="alignment-faking-sft",
            name=cfg.training.run_name,
            config=OmegaConf.to_container(cfg, resolve=True)
        )
    
    # Create output directory
    os.makedirs(cfg.training.output_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(cfg.training.output_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)
    
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(cfg.model)
    
    # Ensure device consistency
    model = ensure_model_device_consistency(model)
    
    logger.info("Preparing datasets...")
    train_dataset, val_dataset = prepare_data(cfg.data, tokenizer)
    
    # Data collator
    data_collator = DeviceAwareDataCollator(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=cfg.training.output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_steps=cfg.training.warmup_steps,
        logging_steps=cfg.training.logging_steps,
        save_steps=cfg.training.save_steps,
        eval_steps=cfg.training.eval_steps,
        eval_strategy=cfg.training.eval_strategy,
        save_strategy=cfg.training.save_strategy,
        load_best_model_at_end=cfg.training.load_best_model_at_end,
        metric_for_best_model=cfg.training.metric_for_best_model,
        greater_is_better=cfg.training.greater_is_better,
        fp16=cfg.training.fp16,
        dataloader_pin_memory=cfg.training.dataloader_pin_memory,
        remove_unused_columns=cfg.training.remove_unused_columns,
        report_to=cfg.training.report_to if cfg.use_wandb else "none",
        run_name=cfg.training.run_name,
        save_total_limit=3,
        logging_dir=os.path.join(cfg.training.output_dir, "logs"),
        gradient_checkpointing=True,  # Enable gradient checkpointing
        ddp_find_unused_parameters=False,  # Fix for multi-GPU training
        dataloader_num_workers=0,  # Avoid multiprocessing issues
        # Multi-GPU specific settings
        deepspeed=None,  # Disable deepspeed for now
        local_rank=-1,  # Let accelerate handle this
        # Fix for device mismatch in loss computation
        dataloader_drop_last=True,  # Ensure consistent batch sizes
    )
    
    # Initialize trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=cfg.training.early_stopping_patience, 
                early_stopping_threshold=cfg.training.early_stopping_threshold
            ), 
            ValidationLossCallback()
        ]
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    logger.info("Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(cfg.training.output_dir)
    
    logger.info("Training completed!")
    
    if cfg.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
