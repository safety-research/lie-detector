"""
TTPD (Training of Truth and Polarity Direction) classifier for robust lie detection.

Reference: "Truth is Universal: Robust Detection of Lies in LLMs" 
by Lennart Bürger, Fred A. Hamprecht, Boaz Nadler (2024)
https://arxiv.org/abs/...

TTPD addresses the failure of linear probes to generalize from affirmative to 
negated statements by identifying a two-dimensional truth subspace with distinct
general truth (tG) and polarity-sensitive truth (tP) directions.
"""

from typing import List, Optional, Tuple
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..core.base_detector import BaseLieDetector


class TTPDClassifier(BaseLieDetector):
    """
    TTPD classifier using 2D truth subspace with general + polarity directions.
    
    Based on Bürger et al. (2024), this method:
    1. Identifies a 2D truth subspace in model representations
    2. Extracts general truth direction (tG) and polarity direction (tP)
    3. Uses both dimensions for robust lie detection across statement types
    
    Key insight: Truth is encoded in a 2D subspace rather than single vector,
    explaining why simple probes fail on negated statements.
    """
    
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
                 layer_ids: List[int] = None, token_position: str = "last"):
        """
        Initialize TTPD classifier.
        
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
        
        # Truth subspace components
        self.truth_directions = None  # 2D truth subspace basis [tG, tP]
        self.classifier = LogisticRegression(random_state=42, max_iter=1000)
        self.scaler = StandardScaler()
        
        self.is_fitted = False
        
    def fit(self, statements: List[str], labels: List[bool], contexts: List[str] = None) -> None:
        """
        Fit TTPD classifier by learning 2D truth subspace.
        
        Args:
            statements: Training statements (mix of affirmative/negated)
            labels: True for honest, False for lying
            contexts: Optional contexts for statements
        """
        print(f"Learning 2D truth subspace for {len(statements)} statements...")
        
        # Extract activations
        activations = []
        for i, statement in enumerate(statements):
            context = contexts[i] if contexts else ""
            full_text = f"{context} {statement}".strip()
            
            activation = self._extract_activations(full_text)
            activations.append(activation)
            
        activations = np.vstack(activations)
        
        # Separate by truth value for subspace identification
        honest_activations = activations[[i for i, label in enumerate(labels) if label]]
        lying_activations = activations[[i for i, label in enumerate(labels) if not label]]
        
        # Learn 2D truth subspace
        self.truth_directions = self._learn_truth_subspace(honest_activations, lying_activations)
        
        # Project activations onto truth subspace
        truth_features = self._project_to_truth_subspace(activations)
        
        # Scale features
        truth_features_scaled = self.scaler.fit_transform(truth_features)
        
        # Train classifier on 2D truth features
        y = [0 if label else 1 for label in labels]  # Convert to 0=honest, 1=lying
        self.classifier.fit(truth_features_scaled, y)
        
        self.is_fitted = True
        print(f"TTPD classifier fitted! Truth subspace dimensions: {self.truth_directions.shape}")
        
    def predict(self, statements: List[str], contexts: List[str] = None) -> List[float]:
        """
        Predict lie probabilities using TTPD classifier.
        
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
            
        activations = np.vstack(activations)
        
        # Project to truth subspace
        truth_features = self._project_to_truth_subspace(activations)
        truth_features_scaled = self.scaler.transform(truth_features)
        
        # Get probabilities
        probabilities = self.classifier.predict_proba(truth_features_scaled)[:, 1]
        
        return probabilities.tolist()
        
    def _learn_truth_subspace(self, honest_activations: np.ndarray, 
                             lying_activations: np.ndarray) -> np.ndarray:
        """
        Learn 2D truth subspace from honest vs lying activations.
        
        Args:
            honest_activations: Activations from honest statements
            lying_activations: Activations from lying statements
            
        Returns:
            2D truth subspace basis vectors [tG, tP]
        """
        # Compute difference in means (general truth direction)
        honest_mean = np.mean(honest_activations, axis=0)
        lying_mean = np.mean(lying_activations, axis=0)
        tG = honest_mean - lying_mean
        tG = tG / np.linalg.norm(tG)  # Normalize
        
        # Find polarity direction using PCA on residuals
        # Project out the general truth direction
        all_activations = np.vstack([honest_activations, lying_activations])
        
        # Remove general truth component
        projections = np.dot(all_activations, tG).reshape(-1, 1)
        residuals = all_activations - projections * tG.reshape(1, -1)
        
        # Find principal component of residuals (polarity direction)
        pca = PCA(n_components=1)
        pca.fit(residuals)
        tP = pca.components_[0]
        tP = tP / np.linalg.norm(tP)  # Normalize
        
        # Stack to form 2D subspace
        truth_directions = np.vstack([tG, tP])
        
        return truth_directions
        
    def _project_to_truth_subspace(self, activations: np.ndarray) -> np.ndarray:
        """
        Project activations onto 2D truth subspace.
        
        Args:
            activations: Input activations
            
        Returns:
            2D projections [tG_coord, tP_coord]
        """
        if self.truth_directions is None:
            raise ValueError("Truth directions not learned yet")
            
        # Project onto each truth direction
        projections = np.dot(activations, self.truth_directions.T)
        
        return projections
        
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
                
                # Use last token position as per Marks & Tegmark (2023)
                # They showed LLMs encode truth info about statement above this token
                if self.token_position == "last":
                    seq_len = inputs['attention_mask'].sum().item()
                    activation = layer_output[seq_len - 1]
                elif self.token_position == "mean":
                    mask = inputs['attention_mask'][0].unsqueeze(-1)
                    activation = (layer_output * mask).sum(0) / mask.sum()
                elif self.token_position == "max":
                    activation = layer_output.max(0)[0]
                else:
                    activation = layer_output[-1]  # Default to last
                    
                layer_activations.append(activation.cpu().numpy())
                
        # Concatenate activations from all layers
        if layer_activations:
            return np.concatenate(layer_activations)
        else:
            return np.zeros(self.model.config.hidden_size)
            
    def get_truth_subspace_info(self) -> dict:
        """Get information about the learned truth subspace."""
        if not self.is_fitted or self.truth_directions is None:
            return {}
            
        tG, tP = self.truth_directions[0], self.truth_directions[1]
        
        return {
            'general_truth_norm': float(np.linalg.norm(tG)),
            'polarity_truth_norm': float(np.linalg.norm(tP)),
            'directions_orthogonality': float(np.dot(tG, tP)),  # Should be ~0 if orthogonal
            'subspace_dimension': self.truth_directions.shape[0]
        }
        
    def get_method_name(self) -> str:
        """Return method name."""
        return f"TTPD Classifier (layers {self.layer_ids})"