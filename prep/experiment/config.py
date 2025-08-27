"""
Configuration handling for the bundle module.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class BundleConfig:
    """Configuration for bundling operations."""
    
    dataset_path: str
    model_name: str
    format_type: str = "chat"
    train_folds: list = None
    eval_folds: list = None
    output_dir: str = "./bundled_data"
    max_train_examples: Optional[int] = None
    max_eval_examples: Optional[int] = None
    balance_strategy: str = "downsample"
    val_split: Optional[float] = None
    seed: int = 42
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.train_folds is None:
            self.train_folds = []
        if self.eval_folds is None:
            self.eval_folds = []
        
        # Validate format type
        if self.format_type not in ["chat", "base_transcript"]:
            raise ValueError(f"Invalid format_type: {self.format_type}")
        
        # Validate balance strategy
        if self.balance_strategy not in ["downsample", "upsample"]:
            raise ValueError(f"Invalid balance_strategy: {self.balance_strategy}")
        
        # Validate val_split
        if self.val_split is not None:
            if not 0.0 <= self.val_split <= 0.3:
                raise ValueError("val_split must be between 0.0 and 0.3")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'dataset_path': self.dataset_path,
            'model_name': self.model_name,
            'format_type': self.format_type,
            'train_folds': self.train_folds,
            'eval_folds': self.eval_folds,
            'output_dir': self.output_dir,
            'max_train_examples': self.max_train_examples,
            'max_eval_examples': self.max_eval_examples,
            'balance_strategy': self.balance_strategy,
            'val_split': self.val_split,
            'seed': self.seed
        }
    
    @classmethod
    def from_args(cls, args) -> 'BundleConfig':
        """Create config from argparse arguments."""
        return cls(
            dataset_path=args.dataset,
            model_name=args.model,
            format_type=args.format,
            train_folds=args.train,
            eval_folds=args.eval,
            output_dir=args.output,
            max_train_examples=args.max_train_examples,
            max_eval_examples=args.max_eval_examples,
            balance_strategy=args.balance,
            val_split=args.val_split,
            seed=args.seed
        )