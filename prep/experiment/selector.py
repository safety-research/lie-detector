"""
Fold selection logic for bundling prep datasets.
"""

from pathlib import Path
from typing import Dict, List, Optional
import re
import json


class FoldSelector:
    """Handles discovery and selection of folds from prep datasets."""
    
    def __init__(self, dataset_path: str):
        """
        Initialize with path to prep dataset.
        
        Args:
            dataset_path: Path to prep dataset directory
        """
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
    
    def discover_folds(self) -> Dict[str, List[str]]:
        """
        Discover all available fold types and their categories.
        
        Returns:
            Dictionary mapping fold types to list of fold names
            e.g., {"colors": ["games", "ascii", ...], "why": [...], ...}
        """
        folds = {}
        
        # Check for two possible structures:
        # 1. folds_* directories (original structure)
        # 2. Direct fold directories (prep.dataset structure)
        
        # First, check for folds_* pattern
        fold_dirs = list(self.dataset_path.glob("folds_*"))
        
        if fold_dirs:
            # Original structure with folds_* directories
            for fold_dir in fold_dirs:
                if fold_dir.is_dir():
                    # Extract fold type from directory name
                    fold_type = fold_dir.name.replace("folds_", "")
                    
                    # Find all fold files (train.jsonl, test.jsonl, etc.)
                    fold_names = set()
                    for jsonl_file in fold_dir.glob("*.jsonl"):
                        # Extract fold name from filename
                        fold_name = jsonl_file.stem
                        if fold_name not in ["train", "test"]:  # Skip standard splits
                            fold_names.add(fold_name)
                    
                    # If only train/test exist, this might be a single-fold type
                    if not fold_names and (fold_dir / "train.jsonl").exists():
                        fold_names.add(fold_type)
                    
                    if fold_names:
                        folds[fold_type] = sorted(list(fold_names))
        else:
            # prep.dataset structure - look for directories with train/test files
            fold_names = []
            for item in self.dataset_path.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    # Check if this directory contains train.jsonl or test.jsonl
                    if (item / "train.jsonl").exists() or (item / "test.jsonl").exists():
                        fold_names.append(item.name)
            
            if fold_names:
                # Use "default" as the fold type for this structure
                folds["default"] = sorted(fold_names)
        
        return folds
    
    def select_folds(self, 
                     train_folds: List[str], 
                     eval_folds: Optional[List[str]] = None,
                     fold_type: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Select folds for training and evaluation.
        
        Args:
            train_folds: List of fold names to use for training
            eval_folds: Optional list of fold names for evaluation
                       If not specified, uses remaining folds
            fold_type: Optional fold type to restrict search to
        
        Returns:
            Dictionary with 'train' and 'eval' keys containing selected folds
        
        Raises:
            ValueError: If specified folds are not found or invalid
        """
        available_folds = self.discover_folds()
        
        if not available_folds:
            raise ValueError("No folds found in dataset")
        
        # Find which fold type contains the requested folds
        selected_type = None
        all_available_folds = []
        
        if fold_type:
            if fold_type not in available_folds:
                raise ValueError(f"Fold type '{fold_type}' not found. Available types: {list(available_folds.keys())}")
            selected_type = fold_type
            all_available_folds = available_folds[fold_type]
        else:
            # Search across all fold types
            for ftype, fold_list in available_folds.items():
                # Check if any requested train fold is in this type
                if any(fold in fold_list for fold in train_folds):
                    if selected_type is None:
                        selected_type = ftype
                        all_available_folds = fold_list
                    else:
                        # Check if all requested folds are in the same type
                        if not all(fold in fold_list or fold in all_available_folds 
                                 for fold in train_folds + (eval_folds or [])):
                            raise ValueError("Requested folds span multiple fold types. Please specify --fold-type")
        
        if selected_type is None:
            raise ValueError(f"None of the requested training folds {train_folds} were found in any fold type")
        
        # Validate train folds
        missing_train = [f for f in train_folds if f not in all_available_folds]
        if missing_train:
            raise ValueError(f"Training folds not found: {missing_train}. Available folds for type '{selected_type}': {all_available_folds}")
        
        # Handle eval folds
        if eval_folds is None:
            # Use remaining folds for evaluation
            eval_folds = [f for f in all_available_folds if f not in train_folds]
            if not eval_folds:
                raise ValueError("No folds left for evaluation after selecting training folds")
        else:
            # Validate eval folds
            missing_eval = [f for f in eval_folds if f not in all_available_folds]
            if missing_eval:
                raise ValueError(f"Evaluation folds not found: {missing_eval}. Available folds for type '{selected_type}': {all_available_folds}")
            
            # Check for overlap
            overlap = set(train_folds) & set(eval_folds)
            if overlap:
                raise ValueError(f"Folds cannot be in both train and eval: {overlap}")
        
        # Determine the fold directory based on structure type
        if selected_type == 'default':
            # For direct fold structure, fold_dir is the dataset path itself
            fold_dir = self.dataset_path
        else:
            # For traditional structure, it's folds_<type>
            fold_dir = self.dataset_path / f"folds_{selected_type}"
        
        return {
            'train': train_folds,
            'eval': eval_folds,
            'fold_type': selected_type,
            'fold_dir': fold_dir
        }