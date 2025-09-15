"""
Weights & Biases Integration for Harmfulness Rating Tracking

This module provides integration with W&B to track and compare:
1. Self-reported harmfulness ratings (from bash_session_with_harmfulness)
2. Post-hoc harmfulness ratings (from sequential_harmfulness_scorer)
"""
import os
from typing import Dict, List, Any, Optional
import wandb
import pandas as pd
import numpy as np
from datetime import datetime
import json
from pathlib import Path


class WandbHarmfulnessTracker:
    """Track and compare harmfulness ratings in W&B."""
    
    def __init__(
        self, 
        project_name: str = "self-sycophancy-code-review",
        experiment_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize W&B tracking for harmfulness ratings.
        
        Args:
            project_name: W&B project name
            experiment_name: Name for this experiment run
            config: Configuration dictionary for the experiment
        """
        self.project_name = project_name
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize W&B
        self.run = wandb.init(
            project=project_name,
            name=self.experiment_name,
            config=config or {},
            tags=["harmfulness", "self-sycophancy", "code-review"]
        )
        
        # Define custom charts
        self._define_custom_charts()
        
    def _define_custom_charts(self):
        """Define custom W&B charts for visualization."""
        # Define scatter plot for self vs post-hoc ratings
        wandb.define_metric("comparison/command_index")
        wandb.define_metric("comparison/*", step_metric="comparison/command_index")
        
        # Define histogram metrics
        wandb.define_metric("distribution/*")
        
        # Define difference metrics
        wandb.define_metric("difference/*")
        
    def log_self_reported_rating(
        self, 
        command: str,
        rating: int,
        justification: str,
        action: str,
        timestamp: str,
        instance: Optional[str] = None,
        command_index: Optional[int] = None
    ):
        """
        Log a self-reported harmfulness rating from bash_session.
        
        Args:
            command: The bash command executed
            rating: Self-reported harmfulness rating (1-100)
            justification: Model's justification for the rating
            action: The action type (type_submit, etc.)
            timestamp: When the command was executed
            instance: Instance identifier
            command_index: Index of command in sequence
        """
        wandb.log({
            "self_reported/rating": rating,
            "self_reported/command": wandb.Table(
                data=[[command, rating, justification, action, timestamp]],
                columns=["command", "rating", "justification", "action", "timestamp"]
            ),
            "self_reported/command_index": command_index or 0
        })
        
    def log_post_hoc_rating(
        self,
        command: str,
        prospective_rating: Optional[int] = None,
        prospective_reasoning: Optional[str] = None,
        retrospective_rating: Optional[int] = None,
        retrospective_reasoning: Optional[str] = None,
        rating_model: str = "task_model",
        command_index: Optional[int] = None
    ):
        """
        Log post-hoc harmfulness ratings from the scorer.
        
        Args:
            command: The bash command evaluated
            prospective_rating: Rating before seeing execution
            prospective_reasoning: Reasoning for prospective rating
            retrospective_rating: Rating after seeing execution
            retrospective_reasoning: Reasoning for retrospective rating
            rating_model: Model used for rating
            command_index: Index of command in sequence
        """
        log_data = {
            "post_hoc/rating_model": rating_model,
            "post_hoc/command_index": command_index or 0
        }
        
        if prospective_rating is not None:
            log_data["post_hoc/prospective_rating"] = prospective_rating
            
        if retrospective_rating is not None:
            log_data["post_hoc/retrospective_rating"] = retrospective_rating
            
        # Create table for detailed view
        table_data = [[
            command,
            prospective_rating,
            prospective_reasoning,
            retrospective_rating,
            retrospective_reasoning,
            rating_model
        ]]
        
        log_data["post_hoc/ratings_table"] = wandb.Table(
            data=table_data,
            columns=[
                "command", 
                "prospective_rating", 
                "prospective_reasoning",
                "retrospective_rating",
                "retrospective_reasoning",
                "rating_model"
            ]
        )
        
        wandb.log(log_data)
        
    def log_comparison(
        self,
        command: str,
        self_reported: int,
        post_hoc_prospective: Optional[int] = None,
        post_hoc_retrospective: Optional[int] = None,
        command_index: int = 0
    ):
        """
        Log comparison between self-reported and post-hoc ratings.
        
        Args:
            command: The bash command
            self_reported: Self-reported rating
            post_hoc_prospective: Post-hoc prospective rating
            post_hoc_retrospective: Post-hoc retrospective rating
            command_index: Index of command in sequence
        """
        comparison_data = {
            "comparison/command_index": command_index,
            "comparison/self_reported": self_reported,
            "comparison/command": command
        }
        
        if post_hoc_prospective is not None:
            comparison_data["comparison/post_hoc_prospective"] = post_hoc_prospective
            comparison_data["comparison/diff_prospective"] = post_hoc_prospective - self_reported
            
        if post_hoc_retrospective is not None:
            comparison_data["comparison/post_hoc_retrospective"] = post_hoc_retrospective
            comparison_data["comparison/diff_retrospective"] = post_hoc_retrospective - self_reported
            
        wandb.log(comparison_data)
        
    def log_aggregate_statistics(
        self,
        self_reported_ratings: List[int],
        post_hoc_ratings: Dict[str, List[int]],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log aggregate statistics comparing rating distributions.
        
        Args:
            self_reported_ratings: List of self-reported ratings
            post_hoc_ratings: Dict with 'prospective' and/or 'retrospective' rating lists
            metadata: Additional metadata to log
        """
        stats = {
            "aggregate/num_commands": len(self_reported_ratings),
            "aggregate/self_reported_mean": np.mean(self_reported_ratings),
            "aggregate/self_reported_std": np.std(self_reported_ratings),
            "aggregate/self_reported_median": np.median(self_reported_ratings)
        }
        
        # Calculate statistics for post-hoc ratings
        for rating_type, ratings in post_hoc_ratings.items():
            if ratings:
                stats[f"aggregate/post_hoc_{rating_type}_mean"] = np.mean(ratings)
                stats[f"aggregate/post_hoc_{rating_type}_std"] = np.std(ratings)
                stats[f"aggregate/post_hoc_{rating_type}_median"] = np.median(ratings)
                
                # Calculate differences
                if len(ratings) == len(self_reported_ratings):
                    diffs = [post - self for post, self in zip(ratings, self_reported_ratings)]
                    stats[f"aggregate/diff_{rating_type}_mean"] = np.mean(diffs)
                    stats[f"aggregate/diff_{rating_type}_std"] = np.std(diffs)
                    stats[f"aggregate/underestimation_rate_{rating_type}"] = sum(1 for d in diffs if d > 0) / len(diffs)
                    
        if metadata:
            stats.update({f"aggregate/{k}": v for k, v in metadata.items()})
            
        wandb.log(stats)
        
    def create_rating_distribution_plot(
        self,
        self_reported: List[int],
        post_hoc_prospective: List[int],
        post_hoc_retrospective: List[int]
    ):
        """Create distribution plots for ratings."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Self-reported distribution
        axes[0].hist(self_reported, bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[0].set_title('Self-Reported Ratings')
        axes[0].set_xlabel('Harmfulness Rating')
        axes[0].set_ylabel('Frequency')
        
        # Post-hoc prospective distribution
        if post_hoc_prospective:
            axes[1].hist(post_hoc_prospective, bins=20, alpha=0.7, color='green', edgecolor='black')
            axes[1].set_title('Post-hoc Prospective Ratings')
            axes[1].set_xlabel('Harmfulness Rating')
            
        # Post-hoc retrospective distribution
        if post_hoc_retrospective:
            axes[2].hist(post_hoc_retrospective, bins=20, alpha=0.7, color='red', edgecolor='black')
            axes[2].set_title('Post-hoc Retrospective Ratings')
            axes[2].set_xlabel('Harmfulness Rating')
            
        plt.tight_layout()
        wandb.log({"distribution/ratings_histogram": wandb.Image(fig)})
        plt.close()
        
    def create_scatter_comparison_plot(
        self,
        self_reported: List[int],
        post_hoc: List[int],
        rating_type: str = "prospective"
    ):
        """Create scatter plot comparing self-reported vs post-hoc ratings."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Scatter plot
        ax.scatter(self_reported, post_hoc, alpha=0.6)
        
        # Add diagonal line (perfect agreement)
        max_val = max(max(self_reported), max(post_hoc))
        ax.plot([0, max_val], [0, max_val], 'r--', label='Perfect Agreement')
        
        # Add trend line
        z = np.polyfit(self_reported, post_hoc, 1)
        p = np.poly1d(z)
        ax.plot(self_reported, p(self_reported), 'g-', label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
        
        ax.set_xlabel('Self-Reported Rating')
        ax.set_ylabel(f'Post-hoc {rating_type.capitalize()} Rating')
        ax.set_title(f'Self-Reported vs Post-hoc {rating_type.capitalize()} Ratings')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        wandb.log({f"comparison/scatter_{rating_type}": wandb.Image(fig)})
        plt.close()
        
    def log_sycophancy_metrics(
        self,
        avg_difference: float,
        num_underrated: int,
        num_overrated: int,
        num_same: int,
        is_sycophantic: bool,
        threshold: float
    ):
        """Log sycophancy detection metrics."""
        total = num_underrated + num_overrated + num_same
        
        wandb.log({
            "sycophancy/avg_difference": avg_difference,
            "sycophancy/underestimation_rate": num_underrated / total if total > 0 else 0,
            "sycophancy/overestimation_rate": num_overrated / total if total > 0 else 0,
            "sycophancy/exact_match_rate": num_same / total if total > 0 else 0,
            "sycophancy/is_sycophantic": int(is_sycophantic),
            "sycophancy/threshold": threshold
        })
        
    def finish(self):
        """Finish the W&B run."""
        wandb.finish()


def load_harmfulness_ratings_from_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load harmfulness ratings from JSONL file."""
    ratings = []
    
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    ratings.append(json.loads(line))
                    
    return ratings


def create_wandb_config(task_state) -> Dict[str, Any]:
    model_name = getattr(task_state, "model", "unknown")
    # Convert ModelName object to string if needed
    if hasattr(model_name, '__str__'):
        model_name = str(model_name)

    config = {
        "task": "harmfulness_tracking",
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
    }

    # Also check metadata for non-serializable objects
    if hasattr(task_state, "metadata"):
        # Create a clean copy of metadata with only JSON-serializable values
        clean_metadata = {}
        for key, value in task_state.metadata.items():
            try:
                json.dumps(value)  # Test if serializable
                clean_metadata[key] = value
            except (TypeError, ValueError):
                clean_metadata[key] = str(value)  # Convert to string if not
        config.update(clean_metadata)

    return config