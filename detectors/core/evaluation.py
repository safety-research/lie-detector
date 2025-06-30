"""
Evaluation utilities for lie detection baselines.

Provides standardized metrics and evaluation procedures for comparing
different lie detection approaches on the MASK benchmark.
"""

from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt


class LieDetectionEvaluator:
    """Comprehensive evaluation for lie detection methods."""
    
    def __init__(self):
        """Initialize evaluator."""
        pass
        
    def compute_metrics(self, predictions: List[float], labels: List[bool]) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            predictions: Predicted lie probabilities (0=honest, 1=lying)
            labels: True labels (True=honest, False=lying)
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Convert labels to match predictions (0=honest, 1=lying)
        binary_labels = [0 if label else 1 for label in labels]
        
        # ROC-AUC and PR-AUC
        try:
            roc_auc = roc_auc_score(binary_labels, predictions)
            pr_auc = average_precision_score(binary_labels, predictions)
        except:
            roc_auc = 0.0
            pr_auc = 0.0
            
        # Accuracy at different thresholds
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        threshold_metrics = {}
        
        for threshold in thresholds:
            binary_preds = [1 if p > threshold else 0 for p in predictions]
            accuracy = sum(1 for p, l in zip(binary_preds, binary_labels) if p == l) / len(labels)
            threshold_metrics[f'accuracy_@_{threshold}'] = accuracy
            
        # False positive rate at different recall levels (for MASK benchmark)
        fpr_at_recall = self._compute_fpr_at_recall(predictions, binary_labels)
        
        return {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            **threshold_metrics,
            **fpr_at_recall
        }
        
    def _compute_fpr_at_recall(self, predictions: List[float], labels: List[int]) -> Dict[str, float]:
        """Compute false positive rate at specific recall levels."""
        # Sort by prediction confidence
        sorted_indices = sorted(range(len(predictions)), key=lambda i: predictions[i], reverse=True)
        
        total_positives = sum(labels)
        total_negatives = len(labels) - total_positives
        
        if total_positives == 0 or total_negatives == 0:
            return {'fpr_at_95_recall': 0.0, 'fpr_at_90_recall': 0.0}
            
        true_positives = 0
        false_positives = 0
        fpr_at_recall = {}
        
        for i, idx in enumerate(sorted_indices):
            if labels[idx] == 1:
                true_positives += 1
            else:
                false_positives += 1
                
            recall = true_positives / total_positives
            fpr = false_positives / total_negatives
            
            # Record FPR at target recall levels
            if recall >= 0.95 and 'fpr_at_95_recall' not in fpr_at_recall:
                fpr_at_recall['fpr_at_95_recall'] = fpr
            if recall >= 0.90 and 'fpr_at_90_recall' not in fpr_at_recall:
                fpr_at_recall['fpr_at_90_recall'] = fpr
                
        return fpr_at_recall
        
    def compare_methods(self, results: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple detection methods.
        
        Args:
            results: Dictionary mapping method names to their metric dictionaries
            
        Returns:
            Comparison summary with rankings
        """
        methods = list(results.keys())
        metrics = list(results[methods[0]].keys()) if methods else []
        
        comparison = {}
        
        for metric in metrics:
            values = [(method, results[method][metric]) for method in methods]
            # Sort by metric value (higher is better for most metrics)
            values.sort(key=lambda x: x[1], reverse=True)
            
            comparison[metric] = {
                'ranking': [method for method, _ in values],
                'values': {method: value for method, value in values}
            }
            
        return comparison