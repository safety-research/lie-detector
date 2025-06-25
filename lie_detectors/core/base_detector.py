"""
Base class for all lie detection methods.

Provides common interface for both whitebox and blackbox lie detection approaches
evaluated on the MASK benchmark.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


class BaseLieDetector(ABC):
    """Abstract base class for lie detection methods."""
    
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        """
        Initialize lie detector.
        
        Args:
            model: HuggingFace model to analyze
            tokenizer: Associated tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
    @abstractmethod
    def fit(self, statements: List[str], labels: List[bool], contexts: List[str] = None) -> None:
        """
        Train the lie detector on labeled data.
        
        Args:
            statements: List of statements to analyze
            labels: True for honest, False for deceptive
            contexts: Optional context for each statement
        """
        pass
        
    @abstractmethod
    def predict(self, statements: List[str], contexts: List[str] = None) -> List[float]:
        """
        Predict lie probability for statements.
        
        Args:
            statements: List of statements to analyze
            contexts: Optional context for each statement
            
        Returns:
            List of probabilities (0=honest, 1=lying)
        """
        pass
        
    @abstractmethod
    def get_method_name(self) -> str:
        """Return name of the detection method."""
        pass
        
    def evaluate(self, statements: List[str], labels: List[bool], contexts: List[str] = None) -> Dict[str, float]:
        """
        Evaluate detector performance.
        
        Args:
            statements: Test statements
            labels: True labels (True=honest, False=lying)
            contexts: Optional contexts
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(statements, contexts)
        
        # Convert to binary predictions using 0.5 threshold
        binary_preds = [p > 0.5 for p in predictions]
        
        # Calculate metrics
        tp = sum(1 for p, l in zip(binary_preds, labels) if p and not l)  # Correctly detected lies
        tn = sum(1 for p, l in zip(binary_preds, labels) if not p and l)  # Correctly detected truth
        fp = sum(1 for p, l in zip(binary_preds, labels) if p and l)     # False lie detection
        fn = sum(1 for p, l in zip(binary_preds, labels) if not p and not l)  # Missed lies
        
        accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        }