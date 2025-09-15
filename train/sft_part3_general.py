import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime

import torch
import wandb
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from evaluate import comprehensive_evaluation, save_comprehensive_results, UnifiedEvaluationCallback

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")


# Verify they're loaded
print(f"HF_HOME: {os.getenv('HF_HOME')}")
print(f"TRANSFORMERS_CACHE: {os.getenv('TRANSFORMERS_CACHE')}")


# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# S3 copy before any data processing
from common.utils import copy_s3_folder_to_local  # Assuming this exists, otherwise define below
from train.preprocess_training_data import process_training_data

# Import generalization mappings
try:
    from common.generalization_mappings import generalization_map_1, generalization_map_2
except ImportError as e:
    print(f"Warning: Could not import generalization mappings from common.generalization_mappings: {e}")
    generalization_map_1 = {}
    generalization_map_2 = {}

def prepare_lie_detection_dataset(data, tokenizer, max_length=2048, format_type="base_transcript"):
    """
    Prepare dataset for training based on the format type.
    
    Args:
        data: List of training examples
        tokenizer: The tokenizer to use
        max_length: Maximum sequence length
        format_type: Either "base_transcript" or "llama_chat"
    """
    examples = []
    total_examples = len(data)
    filtered_examples = 0
    
    # Print first few examples to show what the data looks like
    print("\n" + "="*80)
    print("TRAINING DATA EXAMPLES:")
    print("="*80)
    for i in range(min(3, len(data))):
        print(f"\n--- Example {i+1} ---")
        if format_type == "base_transcript":
            print(f"Prompt length: {len(data[i]['prompt'])} characters")
            print(f"Completion: '{data[i]['completion']}'")
            
            # Show first 200 characters of prompt
            prompt_preview = data[i]['prompt'][:200]
            if len(data[i]['prompt']) > 200:
                prompt_preview += "..."
            print(f"Prompt preview: {prompt_preview}")
            
            # Show last 200 characters of prompt
            if len(data[i]['prompt']) > 200:
                prompt_end = data[i]['prompt'][-200:]
                print(f"Prompt end: ...{prompt_end}")
        elif format_type == "llama_chat":
            print(f"Messages: {len(data[i]['messages'])} messages")
            print(f"Completion: '{data[i]['completion']}'")
            for j, msg in enumerate(data[i]['messages']):
                print(f"  Message {j+1} ({msg['role']}): {msg['content'][:100]}...")
        
        print("-" * 40)
    
    for item in data:
        if format_type == "base_transcript":
            # Original format with text prompt
            prompt = item["prompt"]
            completion = item["completion"]
            
            prompt_tokens = tokenizer(
                prompt,
                truncation=True,
                padding=False,
                add_special_tokens=False,
                max_length=max_length-1
            )["input_ids"]
            
            completion_tokens = tokenizer(
                completion,
                truncation=False,
                padding=False,
                add_special_tokens=False
            )["input_ids"]
            
            if len(completion_tokens) != 1:
                filtered_examples += 1
                continue
                
            input_ids = prompt_tokens + completion_tokens
            labels = [-100] * len(prompt_tokens) + completion_tokens
            
        elif format_type == "llama_chat":
            # LLaMA chat format using apply_chat_template
            messages = item["messages"]
            completion = item["completion"]

            # Tokenize the full conversation using the chat template
            conversation_tokens = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=True
            )[0].tolist()

            # Tokenize the completion
            completion_tokens = tokenizer(
                completion,
                truncation=False,
                padding=False,
                add_special_tokens=False
            )["input_ids"]

            input_ids = conversation_tokens + completion_tokens
            labels = [-100] * len(conversation_tokens) + completion_tokens
        
        else:
            raise ValueError(f"Unknown format_type: {format_type}")
        
        examples.append({
            "input_ids": input_ids,
            "labels": labels
        })
    
    print(f"📊 Dataset preparation stats ({format_type} format):")
    print(f"   Total examples: {total_examples}")
    print(f"   Filtered out: {filtered_examples} (completion != 1 token)")
    print(f"   Final examples: {len(examples)}")
    print(f"   Filter rate: {filtered_examples/total_examples:.1%}")
    
    # Print tokenization info for first few examples
    print("\n" + "="*80)
    print("TOKENIZATION EXAMPLES:")
    print("="*80)
    for i in range(min(3, len(examples))):
        print(f"\n--- Tokenization for Example {i+1} ---")
        input_ids = examples[i]["input_ids"]
        labels = examples[i]["labels"]
        print(f"Input IDs length: {len(input_ids)}")
        print(f"Labels length: {len(labels)}")
        print(f"Last 10 input tokens: {input_ids[-10:]}")
        print(f"Decoded last 10 tokens: '{tokenizer.decode(input_ids[-10:])}'")
        print(f"Completion token: {input_ids[-1]} (decoded: '{tokenizer.decode([input_ids[-1]])}')")
    
    return examples

def train_with_kfold_cv_wrapper(config=None, format_type="base_transcript", generalization_map_name=None, k_folds=3):
    """
    Wrapper function that performs proper k-fold cross-validation by calling train_with_config
    with different train/test splits for each fold.
    
    Args:
        config: Training configuration
        format_type: Data format type
        generalization_map_name: Name of generalization map (not used in proper k-fold)
        k_folds: Number of folds for cross-validation (default: 3)
    """
    
    print(f"🔀 Starting proper k-fold cross-validation with {k_folds} folds")
    
    # Load all data
    print("Loading all data...")
    all_data = []
    with open(output_file, 'r') as f:
        for line in f:
            if line.strip():
                all_data.append(json.loads(line))
    
    print(f"Loaded {len(all_data)} total examples")
    
    # Shuffle data for proper k-fold CV
    print("🔀 Shuffling data for k-fold cross-validation...")
    random.seed(42)  # For reproducibility
    random.shuffle(all_data)
    
    # Split data into k folds
    fold_size = len(all_data) // k_folds
    folds = []
    
    for i in range(k_folds):
        start_idx = i * fold_size
        end_idx = start_idx + fold_size if i < k_folds - 1 else len(all_data)
        folds.append(all_data[start_idx:end_idx])
    
    print(f"📊 Data split into {k_folds} folds:")
    for i, fold in enumerate(folds):
        print(f"   Fold {i+1}: {len(fold)} examples")
    
    # Perform k-fold cross-validation
    fold_results = {}
    
    for fold_idx in range(k_folds):
        fold_name = f"fold_{fold_idx + 1}"
        print(f"\n{'='*60}")
        print(f"🎯 Training fold: {fold_name}")
        print(f"{'='*60}")
        
        # Use current fold as test data, all other folds as training data
        test_data = folds[fold_idx]
        train_data = []
        
        for i in range(k_folds):
            if i != fold_idx:
                train_data.extend(folds[i])
        
        print(f"📊 Fold {fold_name} data split:")
        print(f"   Train: {len(train_data)} examples (folds {[j+1 for j in range(k_folds) if j != fold_idx]})")
        print(f"   Test: {len(test_data)} examples (fold {fold_idx + 1})")
        
        if len(train_data) == 0:
            print(f"⚠️  Warning: No training data for fold {fold_name}, skipping...")
            continue
        
        if len(test_data) == 0:
            print(f"⚠️  Warning: No test data for fold {fold_name}, skipping...")
            continue
        
        # Create temporary files for this fold
        fold_dir = f"{run_dir}/fold_{fold_name}"
        os.makedirs(fold_dir, exist_ok=True)
        
        train_file = f"{fold_dir}/train_data.jsonl"
        test_file = f"{fold_dir}/test_data.jsonl"
        
        # Save train and test data to temporary files
        with open(train_file, 'w') as f:
            for item in train_data:
                f.write(json.dumps(item) + '\n')
        
        with open(test_file, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
        
        # Temporarily replace the global output_file with train_file
        original_output_file = output_file
        global output_file
        output_file = train_file
        
        try:
            # For k-fold CV, we need to modify train_with_config to accept train/test data directly
            # Since train_with_config expects to load from output_file, we'll temporarily replace it
            # and modify the function to use the provided train/test split
            
            # Create a modified config that includes fold information
            fold_config = config.copy() if config else {}
            fold_config['fold_name'] = fold_name
            fold_config['train_data'] = train_data
            fold_config['test_data'] = test_data
            
            # Call the original train_with_config function
            fold_accuracy = train_with_config(fold_config, format_type)
            fold_results[fold_name] = fold_accuracy
            
        finally:
            # Restore the original output_file
            output_file = original_output_file
        
        # Clean up temporary files
        import shutil
        shutil.rmtree(fold_dir, ignore_errors=True)
    
    # Log overall results
    if fold_results:
        avg_accuracy = sum(fold_results.values()) / len(fold_results)
        print(f"\n{'='*60}")
        print(f"🎯 K-FOLD CROSS-VALIDATION RESULTS")
        print(f"{'='*60}")
        for fold_name, accuracy in fold_results.items():
            print(f"Fold {fold_name}: {accuracy:.3f}")
        print(f"Average accuracy: {avg_accuracy:.3f}")
        
        # Log to wandb
        wandb.log({
            "kfold/average_accuracy": avg_accuracy,
            "kfold/num_folds": len(fold_results)
        })
        
        for fold_name, accuracy in fold_results.items():
            wandb.log({
                f"kfold/{fold_name}_accuracy": accuracy
            })
    
    return fold_results



def train_with_config(config=None, format_type="base_transcript"):
    """Training function that accepts wandb config for sweeps"""
    
    # Initialize wandb run
    wandb.init(project="lie-detection-llama", config=config)

    config = wandb.config
    
    # Get hyperparameters from config with defaults
    learning_rate = getattr(config, 'learning_rate', 1e-5)
    lora_r = getattr(config, 'lora_r', 16)
    lora_alpha = getattr(config, 'lora_alpha', 32)
    lora_dropout = getattr(config, 'lora_dropout', 0.1)
    per_device_batch_size = getattr(config, 'per_device_batch_size', 8)
    gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 2)
    num_epochs = getattr(config, 'num_epochs', 3)
    lora_dropout = getattr(config, 'lora_dropout', 0.1)
    weight_decay = getattr(config, 'weight_decay', 0.0)
    warmup_ratio = getattr(config, 'warmup_ratio', 0.0)
    
    print(f"🔧 Hyperparameters:")
    print(f"   Learning Rate: {learning_rate}")
    print(f"   LoRA r: {lora_r}, alpha: {lora_alpha}, dropout: {lora_dropout}")
    print(f"   Batch size: {per_device_batch_size}, Epochs: {num_epochs}")
    print(f"   Weight decay: {weight_decay}, Warmup ratio: {warmup_ratio}")
    print(f"   Format type: {format_type}")
    
    # Load model
    model_name = "meta-llama/Meta-Llama-3-8B"
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_cache=False,
    )

    # Add LoRA with sweep parameters
    print("Adding LoRA adapters...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,  # ✅ From sweep
        lora_alpha=lora_alpha,  # ✅ From sweep
        lora_dropout=lora_dropout,  # ✅ From sweep
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Verify trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Load data
    print("Loading dataset...")
    print(f"Loading from file: {output_file}")
    print(f"File exists: {os.path.exists(output_file)}")
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file)
        print(f"File size: {file_size} bytes ({file_size/1024/1024:.2f} MB)")
    
    data = []
    with open(output_file, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    print(f"Loaded {len(data)} examples")
    
    # Check if train_data and test_data are provided (for k-fold CV)
    fold_name = getattr(config, 'fold_name', None) if config else None
    provided_train_data = getattr(config, 'train_data', None) if config else None
    provided_test_data = getattr(config, 'test_data', None) if config else None
    
    if fold_name and provided_train_data is not None and provided_test_data is not None:
        print(f"🎯 K-fold run detected: {fold_name}")
        train_data = provided_train_data
        eval_data = provided_test_data
        
        print(f"📊 K-fold data split:")
        print(f"   Train examples: {len(train_data)}")
        print(f"   Test examples: {len(eval_data)}")
    else:
        # Standard random split for training
        print("🔀 Shuffling data...")
        random.seed(42)  # For reproducibility
        random.shuffle(data)
        
        # Split data
        train_size = int(0.8 * len(data))
        train_data = data[:train_size]
        eval_data = data[train_size:]
        
        print(f"📊 Data split:")
        print(f"   Train examples: {len(train_data)}")
        print(f"   Eval examples: {len(eval_data)}")
        print(f"   Split ratio: {len(train_data)}/{len(eval_data)} ({len(train_data)/len(data):.1%}/{len(eval_data)/len(data):.1%})")
    
    # Print dataset statistics
    print("\n" + "="*80)
    print("DATASET STATISTICS:")
    print("="*80)
    
    # Show completion distribution
    completions = [item['completion'] for item in data]
    a_count = completions.count('A')
    b_count = completions.count('B')
    
    print(f"Completion distribution:")
    print(f"  A (truth): {a_count} ({a_count/len(completions)*100:.1f}%)")
    print(f"  B (lie): {b_count} ({b_count/len(completions)*100:.1f}%)")
    
    # Show prompt length statistics for base_transcript format
    if format_type == "base_transcript":
        prompt_lengths = [len(item['prompt']) for item in data]
        print(f"\nPrompt length statistics:")
        print(f"  Min: {min(prompt_lengths)}, Max: {max(prompt_lengths)}, Avg: {sum(prompt_lengths)/len(prompt_lengths):.1f}")
    
    # Show message statistics for llama_chat format
    elif format_type == "llama_chat":
        message_counts = [len(item['messages']) for item in data]
        print(f"\nMessage count statistics:")
        print(f"  Min: {min(message_counts)}, Max: {max(message_counts)}, Avg: {sum(message_counts)/len(message_counts):.1f}")
    
    # Check class distribution
    original_a_count = sum(1 for item in data if item["completion"] == "A")
    original_b_count = len(data) - original_a_count
    print(f"📊 Data distribution: A={original_a_count} ({original_a_count/len(data):.1%}), B={original_b_count} ({original_b_count/len(data):.1%})")

    # Prepare datasets with format-specific processing
    train_examples = prepare_lie_detection_dataset(train_data, tokenizer, format_type=format_type)
    eval_examples = prepare_lie_detection_dataset(eval_data, tokenizer, format_type=format_type)

    train_dataset = Dataset.from_list(train_examples)
    eval_dataset = Dataset.from_list(eval_examples)

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt"
    )

    #Training arguments with sweep parameters
    run_name = wandb.run.name if wandb.run else f"run-{now_str}"
    training_args = TrainingArguments(
        output_dir=f"{model_output_dir}/sweep-{run_name}",
        per_device_train_batch_size=per_device_batch_size,  # ✅ From sweep
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=num_epochs,  # ✅ From sweep
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        report_to="wandb",
        learning_rate=learning_rate,  # ✅ From sweep
        eval_strategy="steps",
        eval_steps=50,
        bf16=True,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
    )

    # Get token IDs
    a_id = tokenizer("A", add_special_tokens=False)["input_ids"][0]
    b_id = tokenizer("B", add_special_tokens=False)["input_ids"][0]
    print(f"Token IDs: A={a_id}, B={b_id}")

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Add callback
    train_dataloader = trainer.get_train_dataloader()
    unified_callback = UnifiedEvaluationCallback(
            tokenizer=tokenizer,
            a_id=a_id,
            b_id=b_id,
            eval_data=eval_data,
            train_data=train_data[:250],
            train_dataloader=train_dataloader,
            max_batches=500,
            improvement_threshold=0.005
        )
    trainer.add_callback(unified_callback)

    print("Starting training...")
    trainer.train()

    print("\n" + "="*60)
    print("🚀 FINAL COMPREHENSIVE EVALUATION")
    print("="*60)

    # Evaluate on validation set
    val_metrics = comprehensive_evaluation(model, tokenizer, eval_data, "VALIDATION", a_id, b_id)

    wandb.log({
        "final/val_accuracy": val_metrics['accuracy'],
        "final/val_confidence": val_metrics['mean_confidence'],
        "final/val_a_accuracy": val_metrics['a_accuracy'],
        "final/val_b_accuracy": val_metrics['b_accuracy'],
    })

    # Evaluate on training set (optional - might be slow)
    print(f"\n{'='*60}")
    train_metrics = comprehensive_evaluation(model, tokenizer, train_data, "TRAINING", a_id, b_id)
    print(f"🎯 Final Validation Accuracy: {val_metrics['accuracy']:.3f}")

     # Save comprehensive results to WandB run folder
    sweep_id = getattr(wandb.run, 'sweep_id', None) if wandb.run else None
    results_path = save_comprehensive_results(
        config=config,
        val_metrics=val_metrics,
        train_metrics=train_metrics,
        run_name=run_name,
        sweep_id=sweep_id
    )
    
    # Also save a simple summary in our unified run directory
    summary_file = f"{run_dir}/run_summary.json"
    summary = {
        "run_name": run_name,
        "sweep_id": sweep_id,
        "timestamp": now_str,
        "format_type": format_type,
        "val_accuracy": val_metrics['accuracy'],
        "val_confidence": val_metrics['mean_confidence'],
        "train_accuracy": train_metrics['accuracy'],
        "train_confidence": train_metrics['mean_confidence'],
        "data_file": output_file,
        "model_output_dir": f"{model_output_dir}/sweep-{run_name}",
        "wandb_results_path": results_path
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📋 Run summary saved to: {summary_file}")
    
    # ✅ Clean up for next run
    del model, trainer
    torch.cuda.empty_cache()
    
    return val_metrics['accuracy']

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train lie detection model with SFT")
parser.add_argument("--format", type=str, default="base_transcript", 
                   choices=["base_transcript", "llama_chat"],
                   help="Training format type: base_transcript or llama_chat")
parser.add_argument("--sweep", action="store_true", 
                   help="Run hyperparameter sweep instead of single experiment")
parser.add_argument("--s3_source", type=str, 
                   default="s3://dipika-lie-detection-data/processed-data-v4-copy/",
                   help="S3 source for training data")
parser.add_argument("--input_path", type=str, 
                   default=None,
                   help="Local input path for training data (if provided, will use this instead of S3)")
parser.add_argument("--generalization_map_1", action="store_true",
                   help="Use generalization map 1 from common.generalization_mappings for k-fold CV")
parser.add_argument("--generalization_map_2", action="store_true",
                   help="Use generalization map 2 from common.generalization_mappings for k-fold CV")

args = parser.parse_args()

# Determine destination folder with timestamp
now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
run_dir = f"runs/{args.format}/run_{now_str}_{args.format}"
local_data_dir = f"{run_dir}/training_data"
model_output_dir = f"{run_dir}/model_outputs"

# Create the unified run directory
os.makedirs(run_dir, exist_ok=True)
os.makedirs(local_data_dir, exist_ok=True)
os.makedirs(model_output_dir, exist_ok=True)

print(f"📁 Created unified run directory: {run_dir}")
print(f"   ├── {local_data_dir} (training data)")
print(f"   └── {model_output_dir} (model outputs)")

# Use local data if --input_path is provided and exists, otherwise use S3
if args.input_path and os.path.exists(args.input_path) and os.listdir(args.input_path):
    print(f"📁 Using local training data from: {args.input_path}")
    # Copy local data to run directory for consistency
    import shutil
    shutil.copytree(args.input_path, local_data_dir, dirs_exist_ok=True)
    print(f"✅ Local data copied to: {local_data_dir}")
else:
    print(f"📥 Using S3 data source: {args.s3_source}")
    print(f"Copying data from {args.s3_source} to {local_data_dir} ...")
    copy_s3_folder_to_local(args.s3_source, local_data_dir)
    print(f"S3 copy complete. Data available in {local_data_dir}")

# Preprocess the data to create merged training file
print(f"Preprocessing training data with format: {args.format}")
output_file = f"{run_dir}/training_data_{args.format}.jsonl"

# Check if we have generalization map arguments
generalization_map_name = None
if args.generalization_map_1:
    generalization_map_name = "generalization_map_1"
    print(f"Using generalization map 1 from common.generalization_mappings")
elif args.generalization_map_2:
    generalization_map_name = "generalization_map_2"
    print(f"Using generalization map 2 from common.generalization_mappings")

if generalization_map_name:
    process_training_data(local_data_dir, output_file, args.format, generalization_map_name)
else:
    process_training_data(local_data_dir, output_file, args.format)

print(f"Preprocessing complete. Merged file: {output_file}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sweep_config = {
    'method': 'random',  # or 'grid', 'bayes'
    'metric': {
        'name': 'final/val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'values': [1e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3]
        },
        'lora_r': {
            'values': [4, 8, 16, 32, 64, 128, 256]
        },
        'lora_alpha': {
            'values': [8, 16, 32, 64, 128, 256]
        },
        'lora_dropout': {
            'min': 0.0,
            'max': 0.3
        },
        'per_device_batch_size': {
            'values': [4, 8]
        },
        'num_epochs': {
            'values': [3,4,5]
        },
        'lora_dropout': {
            'min': 0.0,
            'max': 0.3
        },
        'weight_decay': {
            'min': 0.0,
            'max': 0.1
        },
        'warmup_ratio': {
            'min': 0.0,
            'max': 0.2
        }
    }
}

if __name__ == "__main__":
    
    if args.sweep:
        # ✅ RUN HYPERPARAMETER SWEEP
        print("🚀 Starting hyperparameter sweep...")
        sweep_id = wandb.sweep(sweep_config, project="lie-detection-llama")
        print(f"📊 Sweep ID: {sweep_id}")
        
        # ✅ FIXED: Remove config parameter from lambda
        wandb.agent(sweep_id, lambda: train_with_config(None, args.format), count=10)
        print("✨ Sweep complete! Check WandB for best hyperparameters.")


        # # ✅ RUN HYPERPARAMETER SWEEP
        # print("🚀 Starting hyperparameter sweep...")
        # sweep_id = wandb.sweep(sweep_config, project="lie-detection-llama")
        # print(f"📊 Sweep ID: {sweep_id}")
        
        # # Run the sweep (10 experiments)
        # wandb.agent(sweep_id, lambda config: train_with_config(config, args.format), count=10)
        # print("✨ Sweep complete! Check WandB for best hyperparameters.")
        
    else:
        # ✅ RUN SINGLE EXPERIMENT OR K-FOLD CV
        print("🚀 Running experiment...")
        
        # Your original single run config
        single_config = {
            "learning_rate": 1e-5,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "per_device_batch_size": 8,
            "gradient_accumulation_steps": 2,
            "num_epochs": 1,
            "lora_dropout": 0.1,
            "weight_decay": 0.0,
            "warmup_ratio": 0.0
        }
        
        # Check if we should run k-fold cross-validation
        if generalization_map_name:
            # Get the generalization map to determine number of folds
            if generalization_map_name == "generalization_map_1":
                generalization_map = generalization_map_1
            elif generalization_map_name == "generalization_map_2":
                generalization_map = generalization_map_2
            else:
                raise ValueError(f"Unknown generalization map: {generalization_map_name}")
            
            num_folds = len(generalization_map)
            print(f"🎯 Running k-fold cross-validation with {num_folds} folds from {generalization_map_name}")
            train_with_kfold_cv_wrapper(single_config, args.format, generalization_map_name, k_folds=num_folds)
        else:
            print("🎯 Running single experiment")
            train_with_config(single_config, args.format)
        
        print("✨ Experiment complete!")

