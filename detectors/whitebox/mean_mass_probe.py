"""
Mean-mass probing for lie detection.

Reference: "The Geometry of Truth: Emergent Linear Structure in Large Language Model 
Representations of True & False Datasets" by Samuel Marks and Max Tegmark (2024)
https://arxiv.org/abs/2310.06824

Mean-mass probing calculates the probe direction as the direct vector between 
the means of true and false statement representations, avoiding optimization 
and providing better handling of colinear features compared to logistic regression.
"""

from typing import List, Optional
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..core.base_detector import BaseLieDetector


class MeanMassProbeDetector(BaseLieDetector):
    """
    Mean-mass probe lie detector using geometric difference between class means.
    
    Based on Marks & Tegmark (2024), this method:
    1. Computes mean representations for honest vs lying statements
    2. Uses the difference vector as the probe direction
    3. Classifies based on projection onto this direction
    
    Advantages over logistic regression:
    - No optimization required (direct calculation)
    - Better handling of colinear features in truth subspace
    - More causally implicated in model decision-making
    """
    
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
                 layer_ids: List[int] = None, token_position: str = "last",
                 use_whitening: bool = False):
        """
        Initialize mean-mass probe detector.
        
        Args:
            model: HuggingFace model to analyze
            tokenizer: Associated tokenizer
            layer_ids: Specific layers to probe (default: middle layers)
            token_position: Which token position to use ("last", "mean", "max")
            use_whitening: Whether to apply whitening transformation
        """
        super().__init__(model, tokenizer)
        
        # Default to middle layers if not specified
        if layer_ids is None:
            n_layers = len(model.transformer.h) if hasattr(model, 'transformer') else len(model.model.layers)
            layer_ids = [n_layers // 2, n_layers - 1]  # Middle and last layer
            
        self.layer_ids = layer_ids
        self.token_position = token_position
        self.use_whitening = use_whitening
        
        # Probe parameters
        self.probe_direction = None  # θ_mm = μ+ - μ-
        self.honest_mean = None      # μ+ (honest statements)
        self.lying_mean = None       # μ- (lying statements)
        self.covariance_inv = None   # Σ^-1 for whitening
        
        self.is_fitted = False
        
    def fit(self, statements: List[str], labels: List[bool], contexts: List[str] = None) -> None:
        """
        Fit mean-mass probe by computing class means.
        
        Args:
            statements: Training statements
            labels: True for honest, False for lying
            contexts: Optional contexts for statements
        """
        print(f"Computing mean-mass probe for {len(statements)} statements...")
        
        # Extract activations for all statements
        honest_activations = []
        lying_activations = []
        
        for i, statement in enumerate(statements):
            context = contexts[i] if contexts else ""
            full_text = f"{context} {statement}".strip()
            
            activation = self._extract_activations(full_text)
            
            if labels[i]:  # Honest
                honest_activations.append(activation)
            else:  # Lying
                lying_activations.append(activation)
                
        # Convert to arrays
        if honest_activations:
            honest_matrix = np.vstack(honest_activations)
            self.honest_mean = np.mean(honest_matrix, axis=0)
        else:
            raise ValueError("No honest examples found in training data")
            
        if lying_activations:
            lying_matrix = np.vstack(lying_activations)
            self.lying_mean = np.mean(lying_matrix, axis=0)
        else:
            raise ValueError("No lying examples found in training data")
            
        # Compute probe direction: θ_mm = μ+ - μ-
        self.probe_direction = self.honest_mean - self.lying_mean
        
        # Compute covariance for whitening if requested
        if self.use_whitening:
            all_activations = np.vstack([honest_matrix, lying_matrix])
            covariance = np.cov(all_activations.T)
            
            # Add small regularization for numerical stability
            covariance += 1e-6 * np.eye(covariance.shape[0])
            
            try:
                self.covariance_inv = np.linalg.inv(covariance)
            except np.linalg.LinAlgError:
                print("Warning: Covariance matrix singular, using pseudo-inverse")
                self.covariance_inv = np.linalg.pinv(covariance)
                
        self.is_fitted = True
        print(f"Mean-mass probe fitted! Direction norm: {np.linalg.norm(self.probe_direction):.4f}")
        
    def predict(self, statements: List[str], contexts: List[str] = None) -> List[float]:
        """
        Predict lie probabilities using mean-mass probe.
        
        Args:
            statements: Statements to evaluate
            contexts: Optional contexts
            
        Returns:
            List of lie probabilities (0=honest, 1=lying)
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")
            
        probabilities = []
        
        for i, statement in enumerate(statements):
            context = contexts[i] if contexts else ""
            full_text = f"{context} {statement}".strip()
            
            activation = self._extract_activations(full_text)
            lie_prob = self._classify_activation(activation)
            probabilities.append(lie_prob)
            
        return probabilities
        
    def _classify_activation(self, activation: np.ndarray) -> float:
        """
        Classify single activation using mean-mass probe.
        
        Args:
            activation: Activation vector to classify
            
        Returns:
            Lie probability (0=honest, 1=lying)
        """
        if self.use_whitening and self.covariance_inv is not None:
            # Apply whitening: p_mm_iid(x) = σ(θ_mm^T Σ^-1 x)
            score = np.dot(self.probe_direction, np.dot(self.covariance_inv, activation))
        else:
            # Standard mean-mass: p_mm(x) = σ(θ_mm^T x)
            score = np.dot(self.probe_direction, activation)
            
        # Apply sigmoid to get probability
        # Positive score -> closer to honest mean -> low lie probability
        # Negative score -> closer to lying mean -> high lie probability
        lie_probability = 1.0 / (1.0 + np.exp(score))
        
        return float(lie_probability)
        
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
            
    def get_probe_info(self) -> dict:
        """Get information about the fitted probe."""
        if not self.is_fitted:
            return {}
            
        return {
            'probe_direction_norm': float(np.linalg.norm(self.probe_direction)),
            'honest_mean_norm': float(np.linalg.norm(self.honest_mean)),
            'lying_mean_norm': float(np.linalg.norm(self.lying_mean)),
            'class_separation': float(np.linalg.norm(self.honest_mean - self.lying_mean)),
            'uses_whitening': self.use_whitening
        }
        
    def get_method_name(self) -> str:
        """Return method name."""
        whitening_str = " + Whitening" if self.use_whitening else ""
        return f"Mean-Mass Probe{whitening_str} (layers {self.layer_ids})"