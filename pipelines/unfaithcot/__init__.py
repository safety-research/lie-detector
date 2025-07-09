"""Chain-of-thought unfaithfulness detection module."""

from .dataset import (
    create_cot_unfaithfulness_dataset,
    create_few_shot_prompt,
    add_token_to_choices,
    TokenType
)

from .solver import (
    cot_unfaithfulness_solver,
    cot_control_solver,
    cot_deceptive_solver
)

from .scorer import (
    cot_faithfulness_scorer,
    cot_answer_accuracy_scorer,
    cot_token_following_scorer,
    cot_combined_scorer,
    check_token_mention,
    extract_answer_letter
)

__all__ = [
    # Dataset utilities
    "create_cot_unfaithfulness_dataset",
    "create_few_shot_prompt",
    "add_token_to_choices",
    "TokenType",
    
    # Solvers
    "cot_unfaithfulness_solver",
    "cot_control_solver",
    "cot_deceptive_solver",
    
    # Scorers
    "cot_faithfulness_scorer",
    "cot_answer_accuracy_scorer", 
    "cot_token_following_scorer",
    "cot_combined_scorer",
    "check_token_mention",
    "extract_answer_letter"
]