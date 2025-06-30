"""
Standard linear probe baseline for lie detection.

Uses logistic regression on LLM activations from specific layers to classify
truthfulness. This serves as the standard whitebox baseline for comparison
with more sophisticated methods.
"""

from typing import List, Optional, Tuple
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..core.base_detector import BaseLieDetector


class LinearProbeDetector(BaseLieDetector):
    """
    Linear probe lie detector using logistic regression on activations.
    
    This is the standard baseline approach that trains a logistic regression
    classifier on hidden layer activations to distinguish between honest
    and deceptive statements. Commonly used across lie detection literature
    as a reference method.
    """
    
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
                 layer_ids: List[int] = None, token_position: str = "last"):
        """
        Initialize linear probe detector.
        
        Args:
            model: HuggingFace model to analyze
            tokenizer: Associated tokenizer
            layer_ids: Specific layers to probe (default: middle layers)
            token_position: Which token position to use ("last", "mean", "max")
        """
        super().__init__(model, tokenizer)
        
        # Default to middle layers if not specified
        if layer_ids is None:
            n_layers = len(model.transformer.h) if hasattr(model, 'transformer') else len(model.model.layers)
            layer_ids = [n_layers // 2, n_layers - 1]  # Middle and last layer
            
        self.layer_ids = layer_ids
        self.token_position = token_position
        
        # Classifier and scaler
        self.classifier = LogisticRegression(random_state=42, max_iter=1000)
        self.scaler = StandardScaler()
        
        self.is_fitted = False
        
    def fit(self, statements: List[str], labels: List[bool], contexts: List[str] = None) -> None:
        """
        Train linear probe on statement activations.
        
        Args:
            statements: Training statements
            labels: True for honest, False for lying
            contexts: Optional contexts for statements
        """
        print(f"Extracting activations for {len(statements)} statements...")
        
        # Extract activations for all statements
        activations = []
        for i, statement in enumerate(statements):
            context = contexts[i] if contexts else ""
            full_text = f"{context} {statement}".strip()
            
            activation = self._extract_activations(full_text)
            activations.append(activation)
            
        # Stack activations
        X = np.vstack(activations)
        y = [0 if label else 1 for label in labels]  # Convert to 0=honest, 1=lying
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train classifier
        self.classifier.fit(X_scaled, y)
        
        self.is_fitted = True
        print(f"Training completed! Feature dimension: {X.shape[1]}")
        
    def predict(self, statements: List[str], contexts: List[str] = None) -> List[float]:
        """
        Predict lie probabilities using linear probe.
        
        Args:
            statements: Statements to evaluate
            contexts: Optional contexts
            
        Returns:
            List of lie probabilities (0=honest, 1=lying)
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")
            
        # Extract activations
        activations = []
        for i, statement in enumerate(statements):
            context = contexts[i] if contexts else ""
            full_text = f"{context} {statement}".strip()
            
            activation = self._extract_activations(full_text)
            activations.append(activation)
            
        # Stack and scale
        X = np.vstack(activations)
        X_scaled = self.scaler.transform(X)
        
        # Get probabilities
        probabilities = self.classifier.predict_proba(X_scaled)[:, 1]
        
        return probabilities.tolist()
        
    def _extract_activations(self, text: str) -> np.ndarray:
        """
        Extract activations from specified layers.
        
        Args:
            text: Input text to process
            
        Returns:
            Concatenated activations from specified layers
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        # Forward pass with activation capture
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            
        # Extract activations from specified layers
        layer_activations = []
        
        for layer_id in self.layer_ids:
            if layer_id < len(hidden_states):
                layer_output = hidden_states[layer_id][0]  # Remove batch dimension
                
                # Select token position
                if self.token_position == "last":
                    # Use last non-padding token
                    seq_len = inputs['attention_mask'].sum().item()
                    activation = layer_output[seq_len - 1]
                elif self.token_position == "mean":
                    # Mean pool over sequence
                    mask = inputs['attention_mask'][0].unsqueeze(-1)
                    activation = (layer_output * mask).sum(0) / mask.sum()
                elif self.token_position == "max":
                    # Max pool over sequence
                    activation = layer_output.max(0)[0]
                else:
                    activation = layer_output[-1]  # Default to last
                    
                layer_activations.append(activation.cpu().numpy())
                
        # Concatenate activations from all layers
        if layer_activations:
            return np.concatenate(layer_activations)
        else:
            # Fallback if no valid layers
            return np.zeros(self.model.config.hidden_size)
            
    def get_method_name(self) -> str:
        """Return method name."""
        return f"Linear Probe (layers {self.layer_ids})"