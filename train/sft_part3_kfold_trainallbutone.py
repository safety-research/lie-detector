import torch
import json
import os
from pathlib import Path
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import numpy as np

import wandb
from datetime import datetime
import dotenv

dotenv.load_dotenv()


def load_jsonl(file_path):
    """Load data from a JSONL file"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def prepare_dataset(data, tokenizer, max_length=1024):  # Default to safer 1024
    """
    Prepare dataset for training using Gemma chat format.
    Expects data format: {"messages": [...], "completion": "A" or "B"}
    """
    examples = []
    
    for item in data:
        if item is None:
            continue
        messages = item["messages"]
        completion = item["completion"]

        try:
            # Use Gemma's chat template to format the conversation
            conversation_tokens = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )[0].tolist()
        except Exception as e:
            print(f"Error formatting messages: {e}")
            continue
        
        # Tokenize completion (should be single token A or B)
        completion_tokens = tokenizer(
            completion,
            truncation=False,
            padding=False,
            add_special_tokens=False
        )["input_ids"]
        
        # Skip if completion is not exactly 1 token
        if len(completion_tokens) != 1:
            continue
        
        # Truncate conversation if too long
        if len(conversation_tokens) >= max_length:
            conversation_tokens = conversation_tokens[:max_length-1]
            
        # Combine tokens
        input_ids = conversation_tokens + completion_tokens
        labels = [-100] * len(conversation_tokens) + completion_tokens
        
        examples.append({
            "input_ids": input_ids,
            "labels": labels
        })
    
    print(f"Prepared {len(examples)} examples from {len(data)} total (filtered out {len(data) - len(examples)})")
    return examples


def evaluate_model_comprehensive(model, tokenizer, eval_data, fold_name):
    """Comprehensive evaluation function with all metrics"""
    model.eval()
    
    predictions = []
    true_labels = []
    probabilities = []
    
    a_id = tokenizer("A", add_special_tokens=False)["input_ids"][0]
    b_id = tokenizer("B", add_special_tokens=False)["input_ids"][0]
    
    print(f"Evaluating {len(eval_data)} examples...")
    
    with torch.no_grad():
        for i, item in enumerate(eval_data):
            if item is None:
                continue
                
            messages = item["messages"]
            true_completion = item["completion"]
            
            try:
                # Format using chat template
                inputs = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                )
                
                # Get model predictions
                outputs = model(inputs)
                logits = outputs.logits[0, -1, :]  # Last token logits
                
                # Get probabilities for A and B tokens
                probs = torch.softmax(logits, dim=-1)
                a_prob = probs[a_id].item()
                b_prob = probs[b_id].item()
                
                # Predict based on higher probability
                predicted = "A" if a_prob > b_prob else "B"
                
                predictions.append(predicted)
                true_labels.append(true_completion)
                probabilities.append({
                    "A_prob": a_prob,
                    "B_prob": b_prob,
                    "predicted": predicted,
                    "true": true_completion
                })
                
                # Progress update
                if (i + 1) % 100 == 0:
                    print(f"Evaluated {i + 1}/{len(eval_data)} examples")
                    
            except Exception as e:
                print(f"Error during evaluation example {i}: {e}")
                continue
    
    if len(predictions) == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
            "classification_report": {},
            "total_examples": 0,
            "failed_examples": len(eval_data)
        }
    
    # Calculate all metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predictions, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        true_labels, predictions, average=None, zero_division=0, labels=["A", "B"]
    )
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions, labels=["A", "B"])
    
    # Classification report
    class_report = classification_report(
        true_labels, predictions, labels=["A", "B"], output_dict=True, zero_division=0
    )
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else [0, 0, 0, 0]
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    results = {
        "fold_name": fold_name,
        "accuracy": float(accuracy),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "precision_per_class": {
            "A": float(precision_per_class[0]) if len(precision_per_class) > 0 else 0,
            "B": float(precision_per_class[1]) if len(precision_per_class) > 1 else 0
        },
        "recall_per_class": {
            "A": float(recall_per_class[0]) if len(recall_per_class) > 0 else 0,
            "B": float(recall_per_class[1]) if len(recall_per_class) > 1 else 0
        },
        "f1_per_class": {
            "A": float(f1_per_class[0]) if len(f1_per_class) > 0 else 0,
            "B": float(f1_per_class[1]) if len(f1_per_class) > 1 else 0
        },
        "support_per_class": {
            "A": int(support_per_class[0]) if len(support_per_class) > 0 else 0,
            "B": int(support_per_class[1]) if len(support_per_class) > 1 else 0
        },
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "labels": ["A", "B"],
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
        },
        "additional_metrics": {
            "specificity": float(specificity),
            "sensitivity": float(recall_per_class[1]) if len(recall_per_class) > 1 else 0,  # Same as recall for positive class
        },
        "classification_report": class_report,
        "total_examples": len(predictions),
        "failed_examples": len(eval_data) - len(predictions),
        "class_distribution": {
            "true": {label: true_labels.count(label) for label in ["A", "B"]},
            "predicted": {label: predictions.count(label) for label in ["A", "B"]}
        }
    }
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS FOR {fold_name.upper()}")
    print(f"{'='*50}")
    print(f"Total examples: {results['total_examples']}")
    print(f"Failed examples: {results['failed_examples']}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Weighted F1: {results['f1_weighted']:.4f}")
    print(f"Weighted Precision: {results['precision_weighted']:.4f}")
    print(f"Weighted Recall: {results['recall_weighted']:.4f}")
    print(f"\nPer-class F1 scores:")
    print(f"  Class A: {results['f1_per_class']['A']:.4f}")
    print(f"  Class B: {results['f1_per_class']['B']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  True\\Pred  A    B")
    print(f"  A        {cm[0,0]:3d}  {cm[0,1]:3d}")
    print(f"  B        {cm[1,0]:3d}  {cm[1,1]:3d}")
    print(f"{'='*50}")
    
    return results





def evaluate_model_comprehensive2(model, tokenizer, eval_data, fold_name, output_dir):
    """Comprehensive evaluation function with all metrics and detailed predictions"""
    model.eval()
    
    predictions = []
    true_labels = []
    detailed_predictions = []  # Store all prediction details
    
    a_id = tokenizer("A", add_special_tokens=False)["input_ids"][0]
    b_id = tokenizer("B", add_special_tokens=False)["input_ids"][0]
    
    print(f"Evaluating {len(eval_data)} examples...")
    
    with torch.no_grad():
        for i, item in enumerate(eval_data):
            if item is None:
                continue
                
            messages = item["messages"]
            true_completion = item["completion"]
            
            try:
                # Format using chat template
                inputs = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                )
                
                # Get model predictions
                outputs = model(inputs)
                logits = outputs.logits[0, -1, :]  # Last token logits
                
                # Get probabilities for A and B tokens
                probs = torch.softmax(logits, dim=-1)
                a_prob = probs[a_id].item()
                b_prob = probs[b_id].item()
                
                # Normalize probabilities (they should sum to 1 for binary case)
                total_prob = a_prob + b_prob
                a_prob_norm = a_prob / total_prob if total_prob > 0 else 0.5
                b_prob_norm = b_prob / total_prob if total_prob > 0 else 0.5
                
                # Predict based on higher probability
                predicted = "A" if a_prob > b_prob else "B"
                confidence = max(a_prob_norm, b_prob_norm)
                
                predictions.append(predicted)
                true_labels.append(true_completion)
                
                # Store detailed prediction info
                detailed_pred = {
                    "example_id": i,
                    "messages": messages,
                    "true_label": true_completion,
                    "predicted_label": predicted,
                    "probabilities": {
                        "A_prob": float(a_prob),
                        "B_prob": float(b_prob),
                        "A_prob_normalized": float(a_prob_norm),
                        "B_prob_normalized": float(b_prob_norm)
                    },
                    "confidence": float(confidence),
                    "correct": predicted == true_completion,
                    "logits": {
                        "A_logit": float(logits[a_id]),
                        "B_logit": float(logits[b_id])
                    }
                }
                detailed_predictions.append(detailed_pred)
                
                # Progress update
                if (i + 1) % 100 == 0:
                    print(f"Evaluated {i + 1}/{len(eval_data)} examples")
                    
            except Exception as e:
                print(f"Error during evaluation example {i}: {e}")
                # Store failed prediction
                detailed_predictions.append({
                    "example_id": i,
                    "messages": messages if 'messages' in locals() else None,
                    "true_label": true_completion if 'true_completion' in locals() else None,
                    "predicted_label": None,
                    "probabilities": None,
                    "confidence": None,
                    "correct": False,
                    "error": str(e)
                })
                continue
    
    # Save detailed predictions to file
    predictions_file = f"{output_dir}/{fold_name}_detailed_predictions.json"
    with open(predictions_file, 'w') as f:
        json.dump({
            "fold_name": fold_name,
            "total_examples": len(eval_data),
            "successful_predictions": len(predictions),
            "failed_predictions": len(eval_data) - len(predictions),
            "predictions": detailed_predictions
        }, f, indent=2)
    
    print(f"Detailed predictions saved to: {predictions_file}")
    
    if len(predictions) == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
            "classification_report": {},
            "total_examples": 0,
            "failed_examples": len(eval_data),
            "predictions_file": predictions_file
        }
    
    # Calculate all metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predictions, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        true_labels, predictions, average=None, zero_division=0, labels=["A", "B"]
    )
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions, labels=["A", "B"])
    
    # Classification report
    class_report = classification_report(
        true_labels, predictions, labels=["A", "B"], output_dict=True, zero_division=0
    )
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else [0, 0, 0, 0]
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Prepare data for ROC curve (convert labels to binary)
    y_true_binary = [1 if label == "B" else 0 for label in true_labels]  # Treat B as positive class
    y_prob_positive = [pred["probabilities"]["B_prob_normalized"] for pred in detailed_predictions if pred["probabilities"]]
    
    results = {
        "fold_name": fold_name,
        "accuracy": float(accuracy),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "precision_per_class": {
            "A": float(precision_per_class[0]) if len(precision_per_class) > 0 else 0,
            "B": float(precision_per_class[1]) if len(precision_per_class) > 1 else 0
        },
        "recall_per_class": {
            "A": float(recall_per_class[0]) if len(recall_per_class) > 0 else 0,
            "B": float(recall_per_class[1]) if len(recall_per_class) > 1 else 0
        },
        "f1_per_class": {
            "A": float(f1_per_class[0]) if len(f1_per_class) > 0 else 0,
            "B": float(f1_per_class[1]) if len(f1_per_class) > 1 else 0
        },
        "support_per_class": {
            "A": int(support_per_class[0]) if len(support_per_class) > 0 else 0,
            "B": int(support_per_class[1]) if len(support_per_class) > 1 else 0
        },
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "labels": ["A", "B"],
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
        },
        "additional_metrics": {
            "specificity": float(specificity),
            "sensitivity": float(recall_per_class[1]) if len(recall_per_class) > 1 else 0,
        },
        "classification_report": class_report,
        "total_examples": len(predictions),
        "failed_examples": len(eval_data) - len(predictions),
        "class_distribution": {
            "true": {label: true_labels.count(label) for label in ["A", "B"]},
            "predicted": {label: predictions.count(label) for label in ["A", "B"]}
        },
        "predictions_file": predictions_file,
        "roc_data": {
            "y_true": y_true_binary,
            "y_prob": y_prob_positive
        } if len(y_prob_positive) == len(y_true_binary) else None
    }
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS FOR {fold_name.upper()}")
    print(f"{'='*50}")
    print(f"Total examples: {results['total_examples']}")
    print(f"Failed examples: {results['failed_examples']}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Weighted F1: {results['f1_weighted']:.4f}")
    print(f"Weighted Precision: {results['precision_weighted']:.4f}")
    print(f"Weighted Recall: {results['recall_weighted']:.4f}")
    print(f"\nPer-class F1 scores:")
    print(f"  Class A: {results['f1_per_class']['A']:.4f}")
    print(f"  Class B: {results['f1_per_class']['B']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  True\\Pred  A    B")
    print(f"  A        {cm[0,0]:3d}  {cm[0,1]:3d}")
    print(f"  B        {cm[1,0]:3d}  {cm[1,1]:3d}")
    print(f"Predictions saved to: {predictions_file}")
    print(f"{'='*50}")
    
    return results


def evaluate_model(model, tokenizer, eval_data):
    """Evaluation function for device_map='auto' setup"""
    model.eval()
    correct = 0
    total = 0
    
    a_id = tokenizer("A", add_special_tokens=False)["input_ids"][0]
    b_id = tokenizer("B", add_special_tokens=False)["input_ids"][0]
    
    with torch.no_grad():
        for item in eval_data:
            if item is None:
                continue
                
            messages = item["messages"]
            true_completion = item["completion"]
            
            try:
                # Format using chat template
                inputs = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                )
                # Don't specify device - let model handle it with device_map
                
                # Get model predictions
                outputs = model(inputs)
                logits = outputs.logits[0, -1, :]  # Last token logits
                
                # Get probabilities for A and B tokens
                a_prob = torch.softmax(logits, dim=-1)[a_id].item()
                b_prob = torch.softmax(logits, dim=-1)[b_id].item()
                
                # Predict based on higher probability
                predicted = "A" if a_prob > b_prob else "B"
                
                if predicted == true_completion:
                    correct += 1
                total += 1
                
            except Exception as e:
                print(f"Error during evaluation: {e}")
                continue
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def train_fold(fold_name, train_data, eval_data, config):
    """Train on a single fold using simple multi-GPU with device_map"""
    
    print(f"\n{'='*60}")
    print(f"Training fold: {fold_name}")
    print(f"Train examples: {len(train_data)}")
    print(f"Eval examples: {len(eval_data)}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"{'='*60}")
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    # Initialize tokenizer
    model_name = config['model_name']
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with automatic multi-GPU distribution
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # This is the key change!
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    # Disable KV cache and enable gradient checkpointing
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    
    # Show which GPUs are being used
    print(f"Model device map: {model.hf_device_map}")
    
    # Add LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)
    
    # Show example formatting
    if len(train_data) > 0:
        print("\nExample of chat template formatting:")
        example_messages = train_data[0]["messages"]
        try:
            formatted = tokenizer.apply_chat_template(
                example_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            print(f"Messages: {example_messages}")
            print(f"Formatted: {repr(formatted)}")
        except Exception as e:
            print(f"Error formatting example: {e}")
    
    # Prepare datasets
    train_examples = prepare_dataset(train_data, tokenizer, max_length=config.get('max_length', 1024))
    eval_examples = prepare_dataset(eval_data, tokenizer, max_length=config.get('max_length', 1024))
    
    if len(train_examples) == 0:
        print("No valid training examples, skipping fold")
        return 0.0
    
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
    
    # Training arguments - NO Accelerate, regular Trainer
    training_args = TrainingArguments(
        output_dir=f"{config['output_dir']}/{fold_name}",
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        num_train_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        save_strategy="steps",
        eval_strategy="steps",
        save_steps=10,
        eval_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # Or use "accuracy" if defined
        greater_is_better=False,            # Because lower loss is better
        save_total_limit=2,
        logging_steps=2,
        bf16=True,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        weight_decay=config['weight_decay'],
        warmup_ratio=config['warmup_ratio'],
        report_to=None,
        gradient_checkpointing=True,
        dataloader_drop_last=True,
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
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
    print("Starting training...")
    trainer.train()
    
    # Comprehensive evaluation
    print("Starting comprehensive evaluation...")
    results = evaluate_model_comprehensive2(model, tokenizer, eval_data, fold_name, config['output_dir'])
    
    # Cleanup
    del model, trainer, train_dataset, eval_dataset
    torch.cuda.empty_cache()
    
    return results



def run_kfold_cv(input_path, config):
    """Run k-fold cross-validation with comprehensive metrics"""
    input_path = Path(input_path)
    
    # Find all fold directories
    fold_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    fold_names = [d.name for d in fold_dirs]
    
    print(f"Found {len(fold_names)} folds: {fold_names}")
    
    # Load all data
    all_fold_data = {}
    for fold_dir in fold_dirs:
        fold_name = fold_dir.name
        train_file = fold_dir / "train.jsonl"
        test_file = fold_dir / "test.jsonl"
        
        if not train_file.exists() or not test_file.exists():
            print(f"Warning: Missing files in {fold_name}, skipping")
            continue
            
        train_data = load_jsonl(train_file)
        test_data = load_jsonl(test_file)
        
        all_fold_data[fold_name] = {
            'train': train_data,
            'test': test_data
        }
        
        print(f"Loaded {fold_name}: {len(train_data)} train, {len(test_data)} test")
    
    # Run k-fold CV
    fold_results = {}
    
    for i, eval_fold in enumerate(all_fold_data.keys()):
        print(f"\n{'='*80}")
        print(f"K-FOLD CV: Using {eval_fold} as validation fold ({i+1}/{len(all_fold_data)})")
        print(f"{'='*80}")
        
        # Combine training data from all other folds
        train_data = []
        for fold_name, fold_data in all_fold_data.items():
            if fold_name != eval_fold:
                train_data.extend(fold_data['train'])
        
        # Use test data from validation fold as eval data
        eval_data = all_fold_data[eval_fold]['train']
        
        print(f"Training on {len(train_data)} examples from {len(all_fold_data)-1} folds")
        print(f"Evaluating on {len(eval_data)} examples from {eval_fold}")
        
        # Train this fold and get comprehensive results
        fold_results[eval_fold] = train_fold(eval_fold, train_data, eval_data, config)
        
        print(f"Fold {eval_fold} - Accuracy: {fold_results[eval_fold]['accuracy']:.3f}, F1: {fold_results[eval_fold]['f1_weighted']:.3f}")
    
    # Calculate average metrics across folds
    if fold_results:
        metrics_to_average = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        avg_metrics = {}
        
        for metric in metrics_to_average:
            values = [fold_results[fold][metric] for fold in fold_results if metric in fold_results[fold]]
            avg_metrics[f'avg_{metric}'] = sum(values) / len(values) if values else 0.0
            avg_metrics[f'std_{metric}'] = np.std(values) if len(values) > 1 else 0.0
        
        print(f"\n{'='*80}")
        print(f"K-FOLD CROSS-VALIDATION SUMMARY")
        print(f"{'='*80}")
        for fold_name, results in fold_results.items():
            print(f"{fold_name:25s} - Acc: {results['accuracy']:.3f}, F1: {results['f1_weighted']:.3f}, Prec: {results['precision_weighted']:.3f}, Rec: {results['recall_weighted']:.3f}")
        
        print(f"\n{'Average Metrics:'}")
        print(f"{'Accuracy:':<15} {avg_metrics['avg_accuracy']:.3f} ± {avg_metrics['std_accuracy']:.3f}")
        print(f"{'F1 Score:':<15} {avg_metrics['avg_f1_weighted']:.3f} ± {avg_metrics['std_f1_weighted']:.3f}")
        print(f"{'Precision:':<15} {avg_metrics['avg_precision_weighted']:.3f} ± {avg_metrics['std_precision_weighted']:.3f}")
        print(f"{'Recall:':<15} {avg_metrics['avg_recall_weighted']:.3f} ± {avg_metrics['std_recall_weighted']:.3f}")
        
        return fold_results, avg_metrics
    else:
        print("No valid folds completed!")
        return {}, {}


def main():
    # Set environment variables for better memory management
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    
    # Fixed configuration
    input_path = "/workspace/lie-detector/organized_balanced_training_20250722_135859/openrouter_google_gemma-3-4b-it/"
    input_path = "/workspace/lie-detector/organized_balanced_training_20250722_135859/openrouter_google_gemma-3-4b-it/folds_Why_llama_chat"
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"./outputs/gemma_kfold_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Configuration
    config = {
        'model_name': 'google/gemma-3-4b-it',
        'learning_rate': 2e-4,
        'num_epochs': 5,
        'batch_size': 8,
        'lora_r': 8,
        'lora_alpha': 16,
        'lora_dropout': 0.05,
        'gradient_accumulation_steps': 2,
        'weight_decay': 0.01,
        'warmup_ratio': 0.03,
        'output_dir': output_dir,
        'max_length': 2675,
    }
    
    print("Configuration:¡")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Run k-fold cross-validation
    fold_results, avg_metrics = run_kfold_cv(input_path, config)
    
    # Save comprehensive results
    results = {
        'config': config,
        'fold_results': fold_results,
        'average_metrics': avg_metrics,
        'timestamp': timestamp,
        'input_path': input_path
    }
    
    results_file = f"{output_dir}/comprehensive_kfold_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nComprehensive results saved to: {results_file}")


if __name__ == "__main__":
    main()
