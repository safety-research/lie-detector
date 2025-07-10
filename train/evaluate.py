from transformers import TrainerCallback
import torch
import numpy as np
import logging
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
from datetime import datetime
import os
import json
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torch.utils.data.distributed as dist_data
import torch.distributed as dist

# Import wandb with error handling

import logging
logger = logging.getLogger(__name__)


class UnifiedEvaluationCallback(TrainerCallback):
    """
    Clean, unified callback that:
    1. Evaluates on train/val during training
    2. Tracks best model performance
    3. Saves comprehensive results for best models
    4. Handles wandb gracefully
    """
    
    def __init__(self, tokenizer, a_id, b_id, eval_data, train_data, 
                 train_dataloader=None, max_batches=5, improvement_threshold=0.005):
        self.tokenizer = tokenizer
        self.a_id = a_id
        self.b_id = b_id
        self.eval_data = eval_data
        self.train_data = train_data
        self.train_dataloader = train_dataloader
        self.max_batches = max_batches
        self.improvement_threshold = improvement_threshold
        
        # Track best performance
        self.best_val_accuracy = 0.0
        self.best_val_step = 0
        
    def _get_wandb(self):
        """Safely get wandb module and run"""
        try:
            import wandb
            return wandb, wandb.run
        except ImportError:
            return None, None
    
    def _calculate_quick_accuracy(self, model, dataloader, dataset_name="dataset"):
        """Fast accuracy calculation for callback (limited batches)"""
        model.eval()
        total_samples = 0
        correct_predictions = 0
        a_probs_sum = 0.0
        b_probs_sum = 0.0
        a_wins = 0
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= self.max_batches:
                    break
                
                try:
                    input_ids = batch['input_ids'].to(model.device)
                    labels = batch['labels'].to(model.device)
                    
                    outputs = model(input_ids=input_ids)
                    logits = outputs.logits[:, -1, :]
                    probs = torch.softmax(logits, dim=-1)
                    a_probs = probs[:, self.a_id]
                    b_probs = probs[:, self.b_id]
                    
                    predictions = torch.where(a_probs > b_probs, self.a_id, self.b_id)
                    true_labels = labels[:, -1]
                    valid_mask = true_labels != -100
                    
                    if valid_mask.sum() > 0:
                        valid_predictions = predictions[valid_mask]
                        valid_true_labels = true_labels[valid_mask]
                        valid_a_probs = a_probs[valid_mask]
                        valid_b_probs = b_probs[valid_mask]
                        
                        batch_correct = (valid_predictions == valid_true_labels).sum().item()
                        batch_total = valid_mask.sum().item()
                        
                        correct_predictions += batch_correct
                        total_samples += batch_total
                        a_probs_sum += valid_a_probs.sum().item()
                        b_probs_sum += valid_b_probs.sum().item()
                        a_wins += (valid_a_probs > valid_b_probs).sum().item()
                    
                    # Clean up
                    del outputs, logits, probs, a_probs, b_probs
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.warning(f"Error processing batch {i} in {dataset_name}: {e}")
                    continue
        
        if total_samples > 0:
            return {
                'accuracy': correct_predictions / total_samples,
                'mean_a_prob': a_probs_sum / total_samples,
                'mean_b_prob': b_probs_sum / total_samples,
                'a_wins': a_wins,
                'total_samples': total_samples,
                'a_win_rate': a_wins / total_samples
            }
        return None
    
    def _save_best_model_results(self, model, step):
        """Save comprehensive evaluation for best model"""
        try:
            print(f"\n🔥 NEW BEST MODEL at step {step}! Running comprehensive evaluation...")
            
            # Import here to avoid circular imports
            from evaluate import comprehensive_evaluation
            
            # Run full comprehensive evaluation
            val_metrics = comprehensive_evaluation(model, self.tokenizer, self.eval_data, "VALIDATION", self.a_id, self.b_id)
            train_metrics = comprehensive_evaluation(model, self.tokenizer, self.train_data, "TRAINING", self.a_id, self.b_id)
            
            # Get wandb info
            wandb_module, wandb_run = self._get_wandb()
            
            # Create results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"best_model_step_{step}_{timestamp}.json"
            
            results = {
                "best_model_info": {
                    "step": step,
                    "val_accuracy": val_metrics['accuracy'],
                    "timestamp": timestamp,
                    "model_name": "meta-llama/Meta-Llama-3-8B",
                    "wandb_run_id": wandb_run.id if wandb_run else None,
                    "wandb_run_name": wandb_run.name if wandb_run else None
                },
                "config": dict(wandb_run.config) if wandb_run else {},
                "validation_metrics": val_metrics,
                "training_metrics": train_metrics,
                "performance_summary": {
                    "val_accuracy": val_metrics['accuracy'],
                    "train_accuracy": train_metrics['accuracy'],
                    "overfitting_gap": train_metrics['accuracy'] - val_metrics['accuracy'],
                    "val_macro_f1": val_metrics['macro_f1'],
                    "val_confidence": val_metrics['mean_confidence'],
                }
            }
            
            # Save file
            if wandb_run:
                filepath = os.path.join(wandb_run.dir, filename)
                
                # Save and log to wandb
                with open(filepath, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                
                wandb_module.log({
                    "best_model/step": step,
                    "best_model/val_accuracy": val_metrics['accuracy'],
                    "best_model/train_accuracy": train_metrics['accuracy'],
                    "best_model/overfitting_gap": train_metrics['accuracy'] - val_metrics['accuracy'],
                    "best_model/val_macro_f1": val_metrics['macro_f1'],
                    "best_model/val_confidence": val_metrics['mean_confidence'],
                }, step=step)
                
                print(f"💾 Best model results saved: {filepath}")
            else:
                # Fallback to local file
                with open(filename, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"💾 Best model results saved: {filename}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error saving best model results: {e}")
            return None
    
    def on_evaluate(self, args, state, control, model, eval_dataloader, **kwargs):
        """Main evaluation callback"""
        
        # Quick evaluation for callback
        val_metrics = self._calculate_quick_accuracy(model, eval_dataloader, "validation")
        train_metrics = None
        if self.train_dataloader:
            train_metrics = self._calculate_quick_accuracy(model, self.train_dataloader, "training")
        
        # Build log message
        log_message = f"Step {state.global_step} Evaluation:"
        
        if val_metrics:
            current_val_accuracy = val_metrics['accuracy']
            log_message += f"\n  📊 VALIDATION: Accuracy={current_val_accuracy:.3f} ({val_metrics['a_wins']}/{val_metrics['total_samples']}), "
            log_message += f"P(A)={val_metrics['mean_a_prob']:.3f}, P(B)={val_metrics['mean_b_prob']:.3f}"
            
            # Check if this is the best model
            if current_val_accuracy > self.best_val_accuracy + self.improvement_threshold:
                self._save_best_model_results(model, state.global_step)
                self.best_val_accuracy = current_val_accuracy
                self.best_val_step = state.global_step
                log_message += f"\n  🔥 NEW BEST MODEL! (Previous: {self.best_val_accuracy - current_val_accuracy + self.improvement_threshold:.3f})"
            else:
                log_message += f"\n  📈 Current best: {self.best_val_accuracy:.3f} at step {self.best_val_step}"
        
        if train_metrics:
            log_message += f"\n  📈 TRAINING:   Accuracy={train_metrics['accuracy']:.3f} ({train_metrics['a_wins']}/{train_metrics['total_samples']}), "
            log_message += f"P(A)={train_metrics['mean_a_prob']:.3f}, P(B)={train_metrics['mean_b_prob']:.3f}"
            
            if val_metrics:
                gap = train_metrics['accuracy'] - val_metrics['accuracy']
                log_message += f"\n  ⚠️  Overfitting gap: {gap:.3f}"
        
        logger.info(log_message)
        
        # Log to wandb if available
        wandb_module, wandb_run = self._get_wandb()
        if wandb_module and wandb_run and args.report_to and "wandb" in args.report_to:
            try:
                log_dict = {'eval_step': state.global_step}
                if val_metrics:
                    log_dict.update({
                        'eval/accuracy': val_metrics['accuracy'],
                        'eval/a_prob': val_metrics['mean_a_prob'],
                        'eval/b_prob': val_metrics['mean_b_prob'],
                        'eval/a_win_rate': val_metrics['a_win_rate'],
                        'eval/is_best': val_metrics['accuracy'] == self.best_val_accuracy
                    })
                if train_metrics:
                    log_dict.update({
                        'train/accuracy': train_metrics['accuracy'],
                        'train/a_prob': train_metrics['mean_a_prob'],
                        'train/b_prob': train_metrics['mean_b_prob'],
                        'train/a_win_rate': train_metrics['a_win_rate'],
                    })
                    if val_metrics:
                        log_dict['overfitting/gap'] = train_metrics['accuracy'] - val_metrics['accuracy']
                
                wandb_module.log(log_dict, step=state.global_step)
            except Exception as e:
                logger.warning(f"Error logging to wandb: {e}")
        
        model.train()



class BestModelTrackingCallback(TrainerCallback):
    """Enhanced callback that saves comprehensive evaluation when validation accuracy improves"""
    
    def __init__(self, tokenizer, a_id, b_id, train_dataloader, eval_data, train_data, max_batches=10):
        self.tokenizer = tokenizer
        self.a_id = a_id
        self.b_id = b_id
        self.train_dataloader = train_dataloader
        self.eval_data = eval_data
        self.train_data = train_data
        self.max_batches = max_batches
        
        # Track best performance
        self.best_val_accuracy = 0.0
        self.best_val_step = 0
        self.best_metrics = None
        self.improvement_threshold = 0.005  # Only save if improvement > 0.5%
        
    def calculate_accuracy(self, model, dataloader, dataset_name="dataset"):
        """Calculate accuracy on a given dataloader - same as original"""
        model.eval()
        
        total_samples = 0
        correct_predictions = 0
        a_probs_sum = 0.0
        b_probs_sum = 0.0
        a_wins = 0
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if self.max_batches is not None and i >= self.max_batches:
                    break
                
                try:
                    input_ids = batch['input_ids'].to(model.device)
                    labels = batch['labels'].to(model.device)
                    
                    # Get model outputs
                    outputs = model(input_ids=input_ids)
                    logits = outputs.logits[:, -1, :]
                    
                    # Get probabilities for A and B
                    probs = torch.softmax(logits, dim=-1)
                    a_probs = probs[:, self.a_id]
                    b_probs = probs[:, self.b_id]
                    
                    # Get predictions
                    predictions = torch.where(a_probs > b_probs, self.a_id, self.b_id)
                    
                    # Get true labels
                    true_labels = labels[:, -1]
                    valid_mask = true_labels != -100
                    
                    if valid_mask.sum() > 0:
                        valid_predictions = predictions[valid_mask]
                        valid_true_labels = true_labels[valid_mask]
                        valid_a_probs = a_probs[valid_mask]
                        valid_b_probs = b_probs[valid_mask]
                        
                        # Calculate accuracy
                        batch_correct = (valid_predictions == valid_true_labels).sum().item()
                        batch_total = valid_mask.sum().item()
                        
                        correct_predictions += batch_correct
                        total_samples += batch_total
                        a_probs_sum += valid_a_probs.sum().item()
                        b_probs_sum += valid_b_probs.sum().item()
                        a_wins += (valid_a_probs > valid_b_probs).sum().item()
                    
                    # Clean up
                    del outputs, logits, probs, a_probs, b_probs
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.warning(f"Error processing batch {i} in {dataset_name}: {e}")
                    continue
        
        # Calculate final metrics
        if total_samples > 0:
            accuracy = correct_predictions / total_samples
            mean_a_prob = a_probs_sum / total_samples
            mean_b_prob = b_probs_sum / total_samples
            a_win_rate = a_wins / total_samples
            
            return {
                'accuracy': accuracy,
                'mean_a_prob': mean_a_prob,
                'mean_b_prob': mean_b_prob,
                'a_wins': a_wins,
                'total_samples': total_samples,
                'a_win_rate': a_win_rate
            }
        else:
            return None
    
    def save_best_model_comprehensive_results(self, model, config, step):
        """Save comprehensive evaluation results for the best model"""
        try:
            print(f"\n🔥 NEW BEST MODEL at step {step}! Running comprehensive evaluation...")
            
            # Run comprehensive evaluation on both datasets
            val_metrics = comprehensive_evaluation(model, self.tokenizer, self.eval_data, "VALIDATION", self.a_id, self.b_id)
            train_metrics = comprehensive_evaluation(model, self.tokenizer, self.train_data, "TRAINING", self.a_id, self.b_id)
            
            # Create a special filename for best model results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"best_model_step_{step}_{timestamp}.json"
            
            # Prepare results with best model info
            results = {
                "best_model_info": {
                    "step": step,
                    "val_accuracy": val_metrics['accuracy'],
                    "improvement_from_start": val_metrics['accuracy'] - 0.5,  # Assuming 50% baseline
                    "timestamp": timestamp,
                    "model_name": "meta-llama/Meta-Llama-3-8B",
                    "wandb_run_id": wandb.run.id,
                    "wandb_run_name": wandb.run.name
                },
                "config": dict(config),
                "validation_metrics": val_metrics,
                "training_metrics": train_metrics,
                "performance_summary": {
                    "val_accuracy": val_metrics['accuracy'],
                    "train_accuracy": train_metrics['accuracy'],
                    "overfitting_gap": train_metrics['accuracy'] - val_metrics['accuracy'],
                    "val_macro_f1": val_metrics['macro_f1'],
                    "val_confidence": val_metrics['mean_confidence'],
                    "val_class_balance": {
                        "a_accuracy": val_metrics['a_accuracy'],
                        "b_accuracy": val_metrics['b_accuracy'],
                        "a_precision": val_metrics['a_precision'],
                        "b_precision": val_metrics['b_precision']
                    }
                }
            }
            
            # Save to WandB directory
            wandb_run_dir = wandb.run.dir
            filepath = os.path.join(wandb_run_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Log special metrics for best model
            wandb.log({
                f"best_model/step": step,
                f"best_model/val_accuracy": val_metrics['accuracy'],
                f"best_model/train_accuracy": train_metrics['accuracy'],
                f"best_model/overfitting_gap": train_metrics['accuracy'] - val_metrics['accuracy'],
                f"best_model/val_macro_f1": val_metrics['macro_f1'],
                f"best_model/val_confidence": val_metrics['mean_confidence'],
            }, step=step)
            
            # Save as artifact
            try:
                artifact = wandb.Artifact(f"best-model-step-{step}-{wandb.run.name}", type="best_model_results")
                artifact.add_file(filepath)
                wandb.log_artifact(artifact)
                print(f"💾 Best model results saved: {filepath}")
                print(f"🔗 Saved as artifact: best-model-step-{step}-{wandb.run.name}")
            except Exception as e:
                print(f"⚠️ Could not save as artifact: {e}")
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving best model results: {e}")
            return None
    
    def on_evaluate(self, args, state, control, model, eval_dataloader, **kwargs):
        """Called during evaluation - track best model and save if improved"""
        
        # Calculate validation accuracy (quick version for callback)
        val_metrics = self.calculate_accuracy(model, eval_dataloader, "validation")
        
        # Calculate training accuracy
        train_metrics = None
        if self.train_dataloader is not None:
            train_metrics = self.calculate_accuracy(model, self.train_dataloader, "training")
        
        # Log current results
        log_message = f"Step {state.global_step} Evaluation:"
        
        if val_metrics:
            current_val_accuracy = val_metrics['accuracy']
            log_message += f"\n  📊 VALIDATION: Accuracy={current_val_accuracy:.3f} ({val_metrics['a_wins']}/{val_metrics['total_samples']}), "
            log_message += f"P(A)={val_metrics['mean_a_prob']:.3f}, P(B)={val_metrics['mean_b_prob']:.3f}"
            
            # 🔥 CHECK IF THIS IS THE BEST MODEL SO FAR
            if current_val_accuracy > self.best_val_accuracy + self.improvement_threshold:
                # Save comprehensive results for this best model
                config = wandb.config if wandb.run else {}
                self.save_best_model_comprehensive_results(model, config, state.global_step)
                
                # Update best tracking
                self.best_val_accuracy = current_val_accuracy
                self.best_val_step = state.global_step
                self.best_metrics = val_metrics.copy()
                
                log_message += f"\n  🔥 NEW BEST MODEL! (Previous best: {self.best_val_accuracy - current_val_accuracy + self.improvement_threshold:.3f})"
            else:
                log_message += f"\n  📈 Current best: {self.best_val_accuracy:.3f} at step {self.best_val_step}"
        
        if train_metrics:
            log_message += f"\n  📈 TRAINING:   Accuracy={train_metrics['accuracy']:.3f} ({train_metrics['a_wins']}/{train_metrics['total_samples']}), "
            log_message += f"P(A)={train_metrics['mean_a_prob']:.3f}, P(B)={train_metrics['mean_b_prob']:.3f}"
            
            # Log overfitting gap
            if val_metrics:
                gap = train_metrics['accuracy'] - val_metrics['accuracy']
                log_message += f"\n  ⚠️  Overfitting gap: {gap:.3f}"
        
        logger.info(log_message)
        
        # Log to wandb
        if args.report_to and "wandb" in args.report_to:
            try:
                import wandb
                if wandb.run is not None:
                    log_dict = {'eval_step': state.global_step}
                    if val_metrics:
                        log_dict.update({
                            'eval/accuracy': val_metrics['accuracy'],
                            'eval/a_prob': val_metrics['mean_a_prob'],
                            'eval/b_prob': val_metrics['mean_b_prob'],
                            'eval/a_win_rate': val_metrics['a_win_rate'],
                            'eval/is_best': val_metrics['accuracy'] == self.best_val_accuracy
                        })
                    if train_metrics:
                        log_dict.update({
                            'train/accuracy': train_metrics['accuracy'],
                            'train/a_prob': train_metrics['mean_a_prob'],
                            'train/b_prob': train_metrics['mean_b_prob'],
                            'train/a_win_rate': train_metrics['a_win_rate'],
                        })
                        
                        # Log overfitting metrics
                        if val_metrics:
                            log_dict['overfitting/gap'] = train_metrics['accuracy'] - val_metrics['accuracy']
                    
                    if log_dict:
                        wandb.log(log_dict, step=state.global_step)
            except ImportError:
                pass
        
        model.train()  # Return to training mode


class ComprehensiveAccuracyCallback(TrainerCallback):
    """Calculate accuracy on both train and validation datasets"""
    
    def __init__(self, tokenizer, a_id, b_id, train_dataloader=None, max_batches=10):
        self.tokenizer = tokenizer
        self.a_id = a_id
        self.b_id = b_id
        self.train_dataloader = train_dataloader
        self.max_batches = max_batches  # Limit batches to avoid slowdown
        
    def calculate_accuracy(self, model, dataloader, dataset_name="dataset"):
        """Calculate accuracy on a given dataloader"""
        model.eval()
        
        total_samples = 0
        correct_predictions = 0
        a_probs_sum = 0.0
        b_probs_sum = 0.0
        a_wins = 0
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if self.max_batches is not None and i >= self.max_batches:
                    break
                
                try:
                    input_ids = batch['input_ids'].to(model.device)
                    labels = batch['labels'].to(model.device)
                    
                    # Get model outputs
                    outputs = model(input_ids=input_ids)
                    logits = outputs.logits[:, -1, :]  # Last token predictions
                    
                    # Get probabilities for A and B
                    probs = torch.softmax(logits, dim=-1)
                    a_probs = probs[:, self.a_id]
                    b_probs = probs[:, self.b_id]
                    
                    # Get predictions (A if P(A) > P(B), else B)
                    predictions = torch.where(a_probs > b_probs, self.a_id, self.b_id)
                    
                    # Get true labels (last token of each sequence, excluding -100)
                    true_labels = labels[:, -1]
                    valid_mask = true_labels != -100
                    
                    if valid_mask.sum() > 0:
                        valid_predictions = predictions[valid_mask]
                        valid_true_labels = true_labels[valid_mask]
                        valid_a_probs = a_probs[valid_mask]
                        valid_b_probs = b_probs[valid_mask]
                        
                        # Calculate accuracy
                        batch_correct = (valid_predictions == valid_true_labels).sum().item()
                        batch_total = valid_mask.sum().item()
                        
                        correct_predictions += batch_correct
                        total_samples += batch_total
                        a_probs_sum += valid_a_probs.sum().item()
                        b_probs_sum += valid_b_probs.sum().item()
                        a_wins += (valid_a_probs > valid_b_probs).sum().item()
                    
                    # Clean up
                    del outputs, logits, probs, a_probs, b_probs
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.warning(f"Error processing batch {i} in {dataset_name}: {e}")
                    continue
        
        # Calculate final metrics
        if total_samples > 0:
            accuracy = correct_predictions / total_samples
            mean_a_prob = a_probs_sum / total_samples
            mean_b_prob = b_probs_sum / total_samples
            a_win_rate = a_wins / total_samples
            
            return {
                'accuracy': accuracy,
                'mean_a_prob': mean_a_prob,
                'mean_b_prob': mean_b_prob,
                'a_wins': a_wins,
                'total_samples': total_samples,
                'a_win_rate': a_win_rate
            }
        else:
            return None
    
    def on_evaluate(self, args, state, control, model, eval_dataloader, **kwargs):
        """Called during evaluation - calculate accuracy on both datasets"""
        
        # Calculate validation accuracy
        val_metrics = self.calculate_accuracy(model, eval_dataloader, "validation")
        
        # Calculate training accuracy (if train dataloader provided)
        train_metrics = None
        if self.train_dataloader is not None:
            train_metrics = self.calculate_accuracy(model, self.train_dataloader, "training")
        
        # Log results
        log_message = f"Step {state.global_step} Evaluation:"
        
        if val_metrics:
            log_message += f"\n  📊 VALIDATION: Accuracy={val_metrics['accuracy']:.3f} ({val_metrics['a_wins']}/{val_metrics['total_samples']}), "
            log_message += f"P(A)={val_metrics['mean_a_prob']:.3f}, P(B)={val_metrics['mean_b_prob']:.3f}"
        
        if train_metrics:
            log_message += f"\n  📈 TRAINING:   Accuracy={train_metrics['accuracy']:.3f} ({train_metrics['a_wins']}/{train_metrics['total_samples']}), "
            log_message += f"P(A)={train_metrics['mean_a_prob']:.3f}, P(B)={train_metrics['mean_b_prob']:.3f}"
        
        logger.info(log_message)
        
        # Log to wandb if available
        if args.report_to and "wandb" in args.report_to:
            try:
                import wandb
                if wandb.run is not None:
                    log_dict = {}
                    if val_metrics:
                        log_dict.update({
                            'eval/accuracy': val_metrics['accuracy'],
                            'eval/a_prob': val_metrics['mean_a_prob'],
                            'eval/b_prob': val_metrics['mean_b_prob'],
                            'eval/a_win_rate': val_metrics['a_win_rate'],
                        })
                    if train_metrics:
                        log_dict.update({
                            'train/accuracy': train_metrics['accuracy'],
                            'train/a_prob': train_metrics['mean_a_prob'],
                            'train/b_prob': train_metrics['mean_b_prob'],
                            'train/a_win_rate': train_metrics['a_win_rate'],
                        })
                    
                    if log_dict:
                        wandb.log(log_dict, step=state.global_step)
            except ImportError:
                pass
        
        model.train()  # Return to training mode



class SimplifiedLogProbsCallback(TrainerCallback):
    def __init__(self, tokenizer, a_id, b_id):
        self.tokenizer = tokenizer
        self.a_id = a_id
        self.b_id = b_id
    
    def on_evaluate(self, args, state, control, model, eval_dataloader, **kwargs):
        model.eval()
        sample_count = 0
        a_probs_sum = 0.0
        b_probs_sum = 0.0
        a_wins = 0
        
        with torch.no_grad():
            for i, batch in enumerate(eval_dataloader):
                if i >= 1:
                    break
                
                input_ids = batch['input_ids'].to(model.device)
                outputs = model(input_ids=input_ids)
                logits = outputs.logits[:, -1, :]
                
                probs = torch.softmax(logits, dim=-1)
                a_probs = probs[:, self.a_id]
                b_probs = probs[:, self.b_id]
                
                batch_size = a_probs.size(0)
                a_probs_sum += a_probs.sum().item()
                b_probs_sum += b_probs.sum().item()
                a_wins += (a_probs > b_probs).sum().item()
                sample_count += batch_size
                
                del outputs, logits, probs, a_probs, b_probs
                torch.cuda.empty_cache()
        
        if sample_count > 0:
            mean_a_prob = a_probs_sum / sample_count
            mean_b_prob = b_probs_sum / sample_count
            a_win_rate = a_wins / sample_count
            
            logger.info(f"Step {state.global_step} - "
                       f"A prob: {mean_a_prob:.4f}, "
                       f"B prob: {mean_b_prob:.4f}, "
                       f"A wins: {a_wins}/{sample_count} ({a_win_rate:.3f})")
        
        model.train()




from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import numpy as np

def comprehensive_evaluation(model, tokenizer, data, dataset_name, a_id, b_id):
    """Evaluate model on all samples in a dataset with comprehensive metrics"""
    model.eval()
    
    results = []
    correct = 0
    total = 0
    a_probs = []
    b_probs = []
    y_true = []
    y_pred = []
    
    print(f"\n=== Evaluating on {dataset_name} dataset ({len(data)} samples) ===")
    
    with torch.no_grad():
        for i, item in enumerate(data):
            prompt = item["prompt"]
            true_completion = item["completion"]
            
            # Get predictions
            p = predict_a_b(prompt, tokenizer, model, a_id, b_id)
            predicted = "A" if p["A"] > p["B"] else "B"
            is_correct = predicted == true_completion
            
            results.append({
                'prompt': prompt,
                'true': true_completion,
                'predicted': predicted,
                'correct': is_correct,
                'p_a': p["A"],
                'p_b': p["B"],
                'confidence': max(p["A"], p["B"])
            })
            
            if is_correct:
                correct += 1
            total += 1
            
            a_probs.append(p["A"])
            b_probs.append(p["B"])
            
            # Store for sklearn metrics (convert to numeric)
            y_true.append(0 if true_completion == 'A' else 1)
            y_pred.append(0 if predicted == 'A' else 1)
            
            # Show progress every 50 samples
            if (i + 1) % 50 == 0:
                current_acc = correct / total
                print(f"  Progress: {i+1}/{len(data)} samples, Current accuracy: {current_acc:.3f}")
    
    # Calculate basic metrics
    accuracy = correct / total
    mean_a_prob = np.mean(a_probs)
    mean_b_prob = np.mean(b_probs)
    mean_confidence = np.mean([r['confidence'] for r in results])
    
    # Count predictions and true labels
    a_predictions = sum(1 for r in results if r['predicted'] == 'A')
    b_predictions = sum(1 for r in results if r['predicted'] == 'B')
    true_a = sum(1 for r in results if r['true'] == 'A')
    true_b = sum(1 for r in results if r['true'] == 'B')
    
    # Calculate per-class accuracy
    a_correct = sum(1 for r in results if r['true'] == 'A' and r['correct'])
    b_correct = sum(1 for r in results if r['true'] == 'B' and r['correct'])
    a_accuracy = a_correct / true_a if true_a > 0 else 0
    b_accuracy = b_correct / true_b if true_b > 0 else 0
    
    # Calculate comprehensive classification metrics using sklearn
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    
    # Precision, Recall, F1 for each class and averages
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_np, y_pred_np, average=None, labels=[0, 1], zero_division=0
    )
    
    # Macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    # Micro averages (same as accuracy for binary classification)
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true_np, y_pred_np, average='micro', zero_division=0
    )
    
    # Weighted averages
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true_np, y_pred_np, average='weighted', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true_np, y_pred_np, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # Additional metrics
    specificity_a = tn / (tn + fp) if (tn + fp) > 0 else 0  # True negative rate for A
    specificity_b = tp / (tp + fn) if (tp + fn) > 0 else 0  # True negative rate for B
    
    # Print comprehensive results
    print(f"\n🎯 {dataset_name.upper()} FINAL RESULTS:")
    print(f"  Overall Accuracy: {accuracy:.3f} ({correct}/{total})")
    print(f"  Mean Confidence: {mean_confidence:.3f}")
    print(f"  Mean P(A): {mean_a_prob:.3f}")
    print(f"  Mean P(B): {mean_b_prob:.3f}")
    
    print(f"\n📊 Prediction Distribution:")
    print(f"  Predicted A: {a_predictions} samples ({a_predictions/total:.1%})")
    print(f"  Predicted B: {b_predictions} samples ({b_predictions/total:.1%})")
    
    print(f"\n📈 True Label Distribution:")
    print(f"  True A: {true_a} samples ({true_a/total:.1%})")
    print(f"  True B: {true_b} samples ({true_b/total:.1%})")
    
    print(f"\n🎯 Per-Class Metrics:")
    print(f"  Class A (label=0):")
    print(f"    Accuracy: {a_accuracy:.3f} ({a_correct}/{true_a})")
    print(f"    Precision: {precision[0]:.3f}")
    print(f"    Recall: {recall[0]:.3f}")
    print(f"    F1-Score: {f1[0]:.3f}")
    print(f"    Specificity: {specificity_a:.3f}")
    print(f"    Support: {support[0]}")
    
    print(f"  Class B (label=1):")
    print(f"    Accuracy: {b_accuracy:.3f} ({b_correct}/{true_b})")
    print(f"    Precision: {precision[1]:.3f}")
    print(f"    Recall: {recall[1]:.3f}")
    print(f"    F1-Score: {f1[1]:.3f}")
    print(f"    Specificity: {specificity_b:.3f}")
    print(f"    Support: {support[1]}")
    
    print(f"\n📊 Aggregate Metrics:")
    print(f"  Macro Averages:")
    print(f"    Precision: {macro_precision:.3f}")
    print(f"    Recall: {macro_recall:.3f}")
    print(f"    F1-Score: {macro_f1:.3f}")
    
    print(f"  Micro Averages:")
    print(f"    Precision: {micro_precision:.3f}")
    print(f"    Recall: {micro_recall:.3f}")
    print(f"    F1-Score: {micro_f1:.3f}")
    
    print(f"  Weighted Averages:")
    print(f"    Precision: {weighted_precision:.3f}")
    print(f"    Recall: {weighted_recall:.3f}")
    print(f"    F1-Score: {weighted_f1:.3f}")
    
    print(f"\n📈 Confusion Matrix:")
    print(f"           Predicted")
    print(f"         A(0)  B(1)")
    print(f"True A(0) {tn:3d}  {fp:3d}")
    print(f"     B(1) {fn:3d}  {tp:3d}")
    
    # Show worst predictions (lowest confidence incorrect predictions)
    incorrect = [r for r in results if not r['correct']]
    if incorrect:
        worst = sorted(incorrect, key=lambda x: x['confidence'])[:3]
        print(f"\n❌ Worst Predictions (lowest confidence errors):")
        for i, r in enumerate(worst, 1):
            print(f"  {i}. True: {r['true']}, Predicted: {r['predicted']} (conf: {r['confidence']:.3f})")
            print(f"     Prompt: {r['prompt'][:80]}...")
    
    # Show best predictions (highest confidence correct predictions)
    correct_results = [r for r in results if r['correct']]
    if correct_results:
        best = sorted(correct_results, key=lambda x: x['confidence'], reverse=True)[:3]
        print(f"\n✅ Best Predictions (highest confidence correct):")
        for i, r in enumerate(best, 1):
            print(f"  {i}. True: {r['true']}, Predicted: {r['predicted']} (conf: {r['confidence']:.3f})")
            print(f"     Prompt: {r['prompt'][:80]}...")
    
    # Print sklearn classification report for additional insights
    print(f"\n📋 Classification Report:")
    target_names = ['A', 'B']
    print(classification_report(y_true_np, y_pred_np, target_names=target_names, digits=3))
    
    # Return comprehensive metrics dictionary
    return {
        # Basic metrics
        'accuracy': accuracy,
        'total_samples': total,
        'correct': correct,
        'mean_a_prob': mean_a_prob,
        'mean_b_prob': mean_b_prob,
        'mean_confidence': mean_confidence,
        
        # Per-class metrics
        'a_accuracy': a_accuracy,
        'b_accuracy': b_accuracy,
        'a_precision': precision[0],
        'a_recall': recall[0],
        'a_f1': f1[0],
        'a_specificity': specificity_a,
        'a_support': support[0],
        'b_precision': precision[1],
        'b_recall': recall[1],
        'b_f1': f1[1],
        'b_specificity': specificity_b,
        'b_support': support[1],
        
        # Macro metrics
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        
        # Micro metrics
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        
        # Weighted metrics
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        
        # Confusion matrix components
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp,
        'confusion_matrix': cm.tolist(),
        
        # Distribution metrics
        'a_predictions': a_predictions,
        'b_predictions': b_predictions,
        'true_a': true_a,
        'true_b': true_b,
        
        # Raw results for further analysis
        'results': results,
        'y_true': y_true,
        'y_pred': y_pred
    }
# def comprehensive_evaluation(model, tokenizer, data, dataset_name, a_id, b_id):
#     """Evaluate model on all samples in a dataset"""
#     model.eval()
    
#     results = []
#     correct = 0
#     total = 0
#     a_probs = []
#     b_probs = []
    
#     print(f"\n=== Evaluating on {dataset_name} dataset ({len(data)} samples) ===")
    
#     with torch.no_grad():
#         for i, item in enumerate(data):
#             prompt = item["prompt"]
#             true_completion = item["completion"]
            
#             # Get predictions
#             p = predict_a_b(prompt, tokenizer, model, a_id, b_id)
#             predicted = "A" if p["A"] > p["B"] else "B"
#             is_correct = predicted == true_completion
            
#             results.append({
#                 'prompt': prompt,
#                 'true': true_completion,
#                 'predicted': predicted,
#                 'correct': is_correct,
#                 'p_a': p["A"],
#                 'p_b': p["B"],
#                 'confidence': max(p["A"], p["B"])
#             })
            
#             if is_correct:
#                 correct += 1
#             total += 1
            
#             a_probs.append(p["A"])
#             b_probs.append(p["B"])
            
#             # Show progress every 50 samples
#             if (i + 1) % 50 == 0:
#                 current_acc = correct / total
#                 print(f"  Progress: {i+1}/{len(data)} samples, Current accuracy: {current_acc:.3f}")
    
#     # Calculate comprehensive metrics
#     accuracy = correct / total
#     mean_a_prob = np.mean(a_probs)
#     mean_b_prob = np.mean(b_probs)
#     mean_confidence = np.mean([r['confidence'] for r in results])
    
#     # Count predictions
#     a_predictions = sum(1 for r in results if r['predicted'] == 'A')
#     b_predictions = sum(1 for r in results if r['predicted'] == 'B')
    
#     # Count true labels
#     true_a = sum(1 for r in results if r['true'] == 'A')
#     true_b = sum(1 for r in results if r['true'] == 'B')
    
#     # Calculate per-class accuracy
#     a_correct = sum(1 for r in results if r['true'] == 'A' and r['correct'])
#     b_correct = sum(1 for r in results if r['true'] == 'B' and r['correct'])
    
#     a_accuracy = a_correct / true_a if true_a > 0 else 0
#     b_accuracy = b_correct / true_b if true_b > 0 else 0
    
#     # Print comprehensive results
#     print(f"\n🎯 {dataset_name.upper()} FINAL RESULTS:")
#     print(f"  Overall Accuracy: {accuracy:.3f} ({correct}/{total})")
#     print(f"  Mean Confidence: {mean_confidence:.3f}")
#     print(f"  Mean P(A): {mean_a_prob:.3f}")
#     print(f"  Mean P(B): {mean_b_prob:.3f}")
#     print(f"\n📊 Prediction Distribution:")
#     print(f"  Predicted A: {a_predictions} samples ({a_predictions/total:.1%})")
#     print(f"  Predicted B: {b_predictions} samples ({b_predictions/total:.1%})")
#     print(f"\n📈 True Label Distribution:")
#     print(f"  True A: {true_a} samples ({true_a/total:.1%})")
#     print(f"  True B: {true_b} samples ({true_b/total:.1%})")
#     print(f"\n🎯 Per-Class Accuracy:")
#     print(f"  A Accuracy: {a_accuracy:.3f} ({a_correct}/{true_a})")
#     print(f"  B Accuracy: {b_accuracy:.3f} ({b_correct}/{true_b})")
    
#     # Show worst predictions (lowest confidence incorrect predictions)
#     incorrect = [r for r in results if not r['correct']]
#     if incorrect:
#         worst = sorted(incorrect, key=lambda x: x['confidence'])[:3]
#         print(f"\n❌ Worst Predictions (lowest confidence errors):")
#         for i, r in enumerate(worst, 1):
#             print(f"  {i}. True: {r['true']}, Predicted: {r['predicted']} (conf: {r['confidence']:.3f})")
#             print(f"     Prompt: {r['prompt'][:80]}...")
    
#     # Show best predictions (highest confidence correct predictions)
#     correct_results = [r for r in results if r['correct']]
#     if correct_results:
#         best = sorted(correct_results, key=lambda x: x['confidence'], reverse=True)[:3]
#         print(f"\n✅ Best Predictions (highest confidence correct):")
#         for i, r in enumerate(best, 1):
#             print(f"  {i}. True: {r['true']}, Predicted: {r['predicted']} (conf: {r['confidence']:.3f})")
#             print(f"     Prompt: {r['prompt'][:80]}...")
    
#     return {
#         'accuracy': accuracy,
#         'total_samples': total,
#         'correct': correct,
#         'mean_a_prob': mean_a_prob,
#         'mean_b_prob': mean_b_prob,
#         'mean_confidence': mean_confidence,
#         'a_accuracy': a_accuracy,
#         'b_accuracy': b_accuracy,
#         'results': results
#     }


def predict_a_b(prompt, tokenizer, model, a_id, b_id):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
        return {
            "A": probs[a_id].item(),
            "B": probs[b_id].item()
        }
        

def save_run_results(config, val_metrics, train_metrics, run_name, sweep_id=None):
    """Save configuration and results to JSON file in WandB run directory"""
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare results dictionary
    results = {
        "run_info": {
            "run_name": run_name,
            "sweep_id": sweep_id,
            "timestamp": timestamp,
            "model_name": "meta-llama/Meta-Llama-3-8B",
            "wandb_run_dir": wandb.run.dir
        },
        "config": dict(config),  # Convert wandb config to dict
        "validation_metrics": val_metrics,
        "training_metrics": train_metrics,
        "summary": {
            "val_accuracy": val_metrics['accuracy'],
            "val_confidence": val_metrics['mean_confidence'],
            "val_a_accuracy": val_metrics['a_accuracy'],
            "val_b_accuracy": val_metrics['b_accuracy'],
            "train_accuracy": train_metrics['accuracy'],
            "train_confidence": train_metrics['mean_confidence'],
        }
    }
    
    # Save in WandB run directory
    wandb_run_dir = wandb.run.dir
    filename = "experiment_results.json"
    filepath = os.path.join(wandb_run_dir, filename)
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)  # default=str handles non-serializable objects
    
    # Also save as WandB artifact for easy access
    try:
        artifact = wandb.Artifact(f"results-{run_name}", type="experiment_results")
        artifact.add_file(filepath)
        wandb.log_artifact(artifact)
        print(f"💾 Results saved to WandB run directory: {filepath}")
        print(f"📊 Results also saved as WandB artifact: results-{run_name}")
    except Exception as e:
        print(f"⚠️  Could not save as artifact: {e}")
        print(f"💾 Results saved to: {filepath}")
    
    return filepath



def save_comprehensive_results(config, val_metrics, train_metrics, run_name, sweep_id=None):
    """Save comprehensive evaluation results to WandB run directory"""
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare comprehensive results dictionary
    results = {
        "run_info": {
            "run_name": run_name,
            "sweep_id": sweep_id,
            "timestamp": timestamp,
            "model_name": "meta-llama/Meta-Llama-3-8B",
            "wandb_run_dir": wandb.run.dir,
            "wandb_run_id": wandb.run.id,
            "wandb_run_url": wandb.run.url
        },
        "config": dict(config),  # Convert wandb config to dict
        "validation_metrics": val_metrics,
        "training_metrics": train_metrics,
        "summary": {
            # Basic metrics
            "val_accuracy": val_metrics['accuracy'],
            "val_confidence": val_metrics['mean_confidence'],
            "train_accuracy": train_metrics['accuracy'],
            "train_confidence": train_metrics['mean_confidence'],
            
            # Per-class performance
            "val_a_accuracy": val_metrics['a_accuracy'],
            "val_b_accuracy": val_metrics['b_accuracy'],
            "val_a_precision": val_metrics['a_precision'],
            "val_b_precision": val_metrics['b_precision'],
            "val_a_recall": val_metrics['a_recall'],
            "val_b_recall": val_metrics['b_recall'],
            "val_a_f1": val_metrics['a_f1'],
            "val_b_f1": val_metrics['b_f1'],
            
            # Aggregate metrics
            "val_macro_precision": val_metrics['macro_precision'],
            "val_macro_recall": val_metrics['macro_recall'],
            "val_macro_f1": val_metrics['macro_f1'],
            "val_micro_precision": val_metrics['micro_precision'],
            "val_micro_recall": val_metrics['micro_recall'],
            "val_micro_f1": val_metrics['micro_f1'],
            "val_weighted_precision": val_metrics['weighted_precision'],
            "val_weighted_recall": val_metrics['weighted_recall'],
            "val_weighted_f1": val_metrics['weighted_f1'],
            
            # Training metrics summary
            "train_a_accuracy": train_metrics['a_accuracy'],
            "train_b_accuracy": train_metrics['b_accuracy'],
            "train_macro_f1": train_metrics['macro_f1'],
            "train_micro_f1": train_metrics['micro_f1'],
        }
    }
    
    # Save in WandB run directory
    wandb_run_dir = wandb.run.dir
    filename = "comprehensive_evaluation_results.json"
    filepath = os.path.join(wandb_run_dir, filename)
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)  # default=str handles non-serializable objects
    
    # Also save a simplified CSV for quick analysis
    csv_filename = "metrics_summary.csv"
    csv_filepath = os.path.join(wandb_run_dir, csv_filename)
    
    # Create CSV with key metrics
    csv_data = [
        "metric,validation,training",
        f"accuracy,{val_metrics['accuracy']:.4f},{train_metrics['accuracy']:.4f}",
        f"macro_f1,{val_metrics['macro_f1']:.4f},{train_metrics['macro_f1']:.4f}",
        f"micro_f1,{val_metrics['micro_f1']:.4f},{train_metrics['micro_f1']:.4f}",
        f"weighted_f1,{val_metrics['weighted_f1']:.4f},{train_metrics['weighted_f1']:.4f}",
        f"a_precision,{val_metrics['a_precision']:.4f},{train_metrics['a_precision']:.4f}",
        f"b_precision,{val_metrics['b_precision']:.4f},{train_metrics['b_precision']:.4f}",
        f"a_recall,{val_metrics['a_recall']:.4f},{train_metrics['a_recall']:.4f}",
        f"b_recall,{val_metrics['b_recall']:.4f},{train_metrics['b_recall']:.4f}",
        f"mean_confidence,{val_metrics['mean_confidence']:.4f},{train_metrics['mean_confidence']:.4f}",
    ]
    
    with open(csv_filepath, 'w') as f:
        f.write('\n'.join(csv_data))
    
    # Log comprehensive metrics to WandB
    wandb_log_dict = {}
    
    # Log validation metrics with prefix
    for key, value in val_metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, list):
            wandb_log_dict[f"final_eval/val_{key}"] = value
    
    # Log training metrics with prefix
    for key, value in train_metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, list):
            wandb_log_dict[f"final_eval/train_{key}"] = value
    
    # Log to WandB
    wandb.log(wandb_log_dict)
    
    # Save as WandB artifacts for easy access
    try:
        # Save comprehensive results as artifact
        artifact = wandb.Artifact(f"comprehensive-results-{run_name}", type="evaluation_results")
        artifact.add_file(filepath)
        artifact.add_file(csv_filepath)
        wandb.log_artifact(artifact)
        
        print(f"💾 Comprehensive results saved to: {filepath}")
        print(f"📊 CSV summary saved to: {csv_filepath}")
        print(f"🔗 Results saved as WandB artifact: comprehensive-results-{run_name}")
        
    except Exception as e:
        print(f"⚠️  Could not save as artifact: {e}")
        print(f"💾 Results saved locally to: {filepath}")
    
    return filepath