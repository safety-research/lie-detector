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
from transformers import EarlyStoppingCallback


dotenv.load_dotenv()


def load_jsonl(file_path):
    """Load data from a JSONL file"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def calculate_max_steps(all_fold_data, scaler=2):
    """Calculate max_steps as scaler * total samples across all folds"""
    total_samples_across_folds = 0
    for fold_name, fold_data in all_fold_data.items():
        total_samples_across_folds += len(fold_data['train']) + len(fold_data['test'])
    
    max_steps = scaler * total_samples_across_folds
    print(f"Total samples across all folds: {total_samples_across_folds}")
    print(f"Calculated max_steps = {scaler} * {total_samples_across_folds} = {max_steps}")
    return max_steps


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


def train_and_evaluate_all(train_fold_name, all_fold_data, config):
    """Train on one fold (train+test data) and evaluate on test data of all folds"""
    
    # Combine train and test data from training fold
    train_data = all_fold_data[train_fold_name]['train'] + all_fold_data[train_fold_name]['test']
    
    print(f"\n{'='*60}")
    print(f"Training on fold: {train_fold_name}")
    print(f"Train examples: {len(train_data)} (train: {len(all_fold_data[train_fold_name]['train'])}, test: {len(all_fold_data[train_fold_name]['test'])})")
    print(f"Will evaluate on test data of all {len(all_fold_data)} folds")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"{'='*60}")
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    run_name = f"train_on_{train_fold_name}_model_{config['model_name']}"
    wandb.init(project="lie-detector", name=run_name, reinit=True)
    wandb.config.update(config)
    run_url = wandb.run.url

    
    # Initialize tokenizer
    model_name = config['model_name']
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with automatic multi-GPU distribution
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
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
    
    # Prepare training dataset
    train_examples = prepare_dataset(train_data, tokenizer, max_length=config.get('max_length', 1024))
    
    if len(train_examples) == 0:
        print("No valid training examples, skipping fold")
        return {}
    
    train_dataset = Dataset.from_list(train_examples)
    
    # Create a small eval dataset from training data for training monitoring
    eval_examples = train_examples[:min(100, len(train_examples))]  # Use first 100 examples for monitoring
    eval_dataset = Dataset.from_list(eval_examples)
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt"
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"{config['output_dir']}/train_on_{train_fold_name}",
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['eval_batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        max_steps=config['max_steps'],
        learning_rate=config['learning_rate'],
        save_strategy="steps",
        eval_strategy="steps",
        save_steps=50,
        eval_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        logging_steps=10,
        bf16=True,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        weight_decay=config['weight_decay'],
        warmup_ratio=config['warmup_ratio'],
        report_to="wandb",
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
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.01
        )]
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    wandb.finish()
    print("Training finished")
    
    # Now evaluate on test data of all folds
    print("Starting evaluation on test data of all folds...")
    fold_results = {}
    
    for fold_name, fold_data in all_fold_data.items():
        print(f"\nEvaluating on fold: {fold_name}")
        
        if fold_name == train_fold_name:
            # For training fold, evaluate on test data (validation check)
            eval_data = fold_data['test']
            eval_type = "validation"
            print(f"Evaluating on {len(eval_data)} validation examples (test data from training fold)")
        else:
            # For other folds, evaluate on both train and test data
            eval_data = fold_data['train'] + fold_data['test']
            eval_type = "test"
            print(f"Evaluating on {len(eval_data)} test examples (train + test data from other fold)")
        
        # Evaluate on this fold's test data
        results = evaluate_model_comprehensive2(
            model, tokenizer, eval_data, 
            f"train_{train_fold_name}_eval_{fold_name}_{eval_type}", 
            config['output_dir']
        )
        
        # Add metadata about evaluation type
        results['eval_type'] = eval_type
        results['is_training_fold'] = (fold_name == train_fold_name)
        
        fold_results[fold_name] = results
        print(f"Fold {fold_name} ({eval_type}) - Accuracy: {results['accuracy']:.3f}, F1: {results['f1_weighted']:.3f}")
    
    # Cleanup
    del model, trainer, train_dataset, eval_dataset
    torch.cuda.empty_cache()
    
    return fold_results, run_url


def run_train_one_eval_all(input_path, config):
    """Train on each fold individually and evaluate on all folds"""
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
    
    # Results for each training fold
    all_training_results = {}
    all_run_urls = {}
    
    # Train on each fold and evaluate on all
    for train_fold_name in all_fold_data.keys():
        print(f"\n{'='*80}")
        print(f"TRAINING ON FOLD: {train_fold_name}")
        print(f"{'='*80}")
        
        # Get combined training data from this fold (train + test)
        train_data_size = len(all_fold_data[train_fold_name]['train'])
        test_data_size = len(all_fold_data[train_fold_name]['test'])
        total_train_size = train_data_size + test_data_size
        
        print(f"Training on {total_train_size} examples from {train_fold_name} (train: {train_data_size}, test: {test_data_size})")
        
        # Train on this fold and evaluate on all folds' test data
        fold_results, run_url = train_and_evaluate_all(
            train_fold_name, all_fold_data, config
        )
        
        all_training_results[train_fold_name] = fold_results
        all_run_urls[train_fold_name] = run_url
        
        # Print summary for this training fold
        print(f"\nSummary for training on {train_fold_name}:")
        for eval_fold, results in fold_results.items():
            eval_type = "validation" if eval_fold == train_fold_name else "test"
            print(f"  Eval on {eval_fold:15s} ({eval_type:10s}) - Acc: {results['accuracy']:.3f}, F1: {results['f1_weighted']:.3f}")
    
    # Calculate overall statistics
    print(f"\n{'='*80}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*80}")
    
    # Separate validation (same fold) and test (different fold) results
    validation_results = {}  # Results on training fold's test data
    test_results = {}        # Results on other folds' test data
    
    for train_fold, fold_results in all_training_results.items():
        for eval_fold, results in fold_results.items():
            if eval_fold == train_fold:
                # Validation result
                validation_results[train_fold] = results
            else:
                # Test result
                if eval_fold not in test_results:
                    test_results[eval_fold] = []
                test_results[eval_fold].append({
                    'train_fold': train_fold,
                    'results': results
                })
    
    # Print validation results
    print("\nValidation Results (training fold's test data):")
    val_accuracies = []
    val_f1_scores = []
    for train_fold, results in validation_results.items():
        acc = results['accuracy']
        f1 = results['f1_weighted']
        val_accuracies.append(acc)
        val_f1_scores.append(f1)
        print(f"  {train_fold:15s} - Acc: {acc:.3f}, F1: {f1:.3f}")
    
    if val_accuracies:
        print(f"  Average validation   - Acc: {np.mean(val_accuracies):.3f}±{np.std(val_accuracies):.3f}, F1: {np.mean(val_f1_scores):.3f}±{np.std(val_f1_scores):.3f}")
    
    # Print test results by evaluation fold
    print("\nTest Results by evaluation fold (across different training folds):")
    all_test_accuracies = []
    all_test_f1_scores = []
    
    for eval_fold, results_list in test_results.items():
        print(f"\nEvaluating on {eval_fold} test data:")
        accuracies = []
        f1_scores = []
        
        for result_info in results_list:
            train_fold = result_info['train_fold']
            results = result_info['results']
            acc = results['accuracy']
            f1 = results['f1_weighted']
            accuracies.append(acc)
            f1_scores.append(f1)
            all_test_accuracies.append(acc)
            all_test_f1_scores.append(f1)
            print(f"  Trained on {train_fold:15s} - Acc: {acc:.3f}, F1: {f1:.3f}")
        
        if accuracies:
            print(f"  Average for {eval_fold:10s} - Acc: {np.mean(accuracies):.3f}±{np.std(accuracies):.3f}, F1: {np.mean(f1_scores):.3f}±{np.std(f1_scores):.3f}")
    
    # Calculate grand averages
    if all_test_accuracies:
        print(f"\nGrand averages across all test evaluations:")
        print(f"Test Accuracy: {np.mean(all_test_accuracies):.3f} ± {np.std(all_test_accuracies):.3f}")
        print(f"Test F1 Score: {np.mean(all_test_f1_scores):.3f} ± {np.std(all_test_f1_scores):.3f}")
    
    if val_accuracies and all_test_accuracies:
        print(f"\nComparison:")
        print(f"Validation Performance: Acc {np.mean(val_accuracies):.3f}±{np.std(val_accuracies):.3f}, F1 {np.mean(val_f1_scores):.3f}±{np.std(val_f1_scores):.3f}")
        print(f"Test Performance:       Acc {np.mean(all_test_accuracies):.3f}±{np.std(all_test_accuracies):.3f}, F1 {np.mean(all_test_f1_scores):.3f}±{np.std(all_test_f1_scores):.3f}")
        
        # Check for overfitting
        val_test_acc_diff = np.mean(val_accuracies) - np.mean(all_test_accuracies)
        val_test_f1_diff = np.mean(val_f1_scores) - np.mean(all_test_f1_scores)
        print(f"Validation - Test gap:  Acc {val_test_acc_diff:+.3f}, F1 {val_test_f1_diff:+.3f}")
        
        if val_test_acc_diff > 0.05:  # 5% threshold
            print("⚠️  Possible overfitting detected (validation significantly higher than test)")
        elif val_test_acc_diff < -0.02:  # Models performing better on test than validation
            print("ℹ️  Models generalizing well (test performance similar or better than validation)")
    
    return all_training_results, all_run_urls


def main():
    # Set environment variables for better memory management
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    
    # Base input path
    base_input_path = "/workspace/lie-detector/organized_balanced_training_20250727_201402_cleaned"
    # Get all model subdirectories
    model_dirs = [d for d in os.listdir(base_input_path) if os.path.isdir(os.path.join(base_input_path, d)) and d.startswith('openrouter_google_gemma-3-')]
    
    print(f"Found model directories: {model_dirs}")
    
    # Model name mapping
    model_name_map = {
        'openrouter_google_gemma-3-4b-it': 'google/gemma-3-4b-it',
        'openrouter_google_gemma-3-12b-it': 'google/gemma-3-12b-it', 
        'openrouter_google_gemma-3-27b-it': 'google/gemma-3-27b-it'
    }
    
    all_model_results = {}
    
    # Configuration (defined outside the loop)
    config = {
        'learning_rate': 2e-4,
        'batch_size': 8,
        'eval_batch_size': 16,  # Larger batch size for evaluation
        'lora_r': 8,
        'lora_alpha': 16,
        'lora_dropout': 0.05,
        'gradient_accumulation_steps': 2,
        'weight_decay': 0.01,
        'warmup_ratio': 0.03,
        'max_length': 2675,
    }
    
    # Extract timestamp from the input path instead of generating a new one
    import re
    timestamp_match = re.search(r'(\d{8}_\d{6})', base_input_path)
    if timestamp_match:
        timestamp = timestamp_match.group(1)
        print(f"Using timestamp from input path: {timestamp}")
    else:
        # Fallback to current timestamp if not found in path
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        print(f"Timestamp not found in input path, using current time: {timestamp}")
    
    for model_dir in model_dirs:
        print(f"\n{'='*80}")
        print(f"Processing model: {model_dir}")
        print(f"{'='*80}")
        
        # Construct the folds directory path based on whether there's a _train_[number] suffix
        if "_train_" in base_input_path:
            import re
            train_match = re.search(r'_train_\d+', base_input_path)
            if train_match:
                train_suffix = train_match.group()
                folds_dir = f"folds_colors_chat_format{train_suffix}"
            else:
                folds_dir = "folds_colors_chat_format"
        else:
            folds_dir = "folds_colors_chat_format"
        
        input_path = os.path.join(base_input_path, model_dir, folds_dir)
        
        # Calculate max_steps based on the fold data
        input_path_obj = Path(input_path)
        fold_dirs = [d for d in input_path_obj.iterdir() if d.is_dir()]
        all_fold_data = {}
        for fold_dir in fold_dirs:
            fold_name = fold_dir.name
            train_file = fold_dir / "train.jsonl"
            test_file = fold_dir / "test.jsonl"
            
            if train_file.exists() and test_file.exists():
                train_data = load_jsonl(train_file)
                test_data = load_jsonl(test_file)
                all_fold_data[fold_name] = {
                    'train': train_data,
                    'test': test_data
                }
        
        max_steps = calculate_max_steps(all_fold_data)
        
        # Create output directory
        output_dir = f"./outputs/{timestamp}/lr_{config['learning_rate']}/{model_dir}_train_one_eval_all_{timestamp}"
        
        # Check if base input path contains _train_[number] pattern
        if "_train_" in base_input_path:
            import re
            train_match = re.search(r'_train_\d+', base_input_path)
            if train_match:
                train_suffix = train_match.group()
                output_dir = f"./outputs/{timestamp}{train_suffix}/lr_{config['learning_rate']}/{model_dir}_train_one_eval_all_{timestamp}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Update config with model-specific values
        config.update({
            'model_name': model_name_map[model_dir],
            'output_dir': output_dir,
            'max_steps': max_steps,
        })
        
        print("Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        # Run train-on-one, evaluate-on-all
        all_results, all_run_urls = run_train_one_eval_all(input_path, config)
        
        # Save comprehensive results
        results = {
            'config': config,
            'all_training_results': all_results,
            'all_run_urls': all_run_urls,
            'timestamp': timestamp,
            'input_path': input_path,
            'method': 'train_one_evaluate_all',
        }
        
        results_file = f"{output_dir}/train_one_eval_all_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        all_model_results[model_dir] = results
        print(f"\nComprehensive results saved to: {results_file}")
    
    # Save combined results
    combined_results_file = f"{output_dir}/all_models_train_one_eval_all_{timestamp}.json"
    with open(combined_results_file, 'w') as f:
        json.dump(all_model_results, f, indent=2)
    
    print(f"\nCombined results saved to: {combined_results_file}")


if __name__ == "__main__":
    main()