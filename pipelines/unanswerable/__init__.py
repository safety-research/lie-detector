"""Unanswerable claims detection module."""

from .dataset import (
    create_unanswerable_claims_dataset,
    extract_question_and_choices,
    add_na_option
)

from .solver import (
    unanswerable_claims_solver,
    generate_unanswerable_question,
    prepare_question_with_na,
    ask_generated_question
)

from .scorer import (
    unanswerable_claims_scorer,
    question_quality_scorer
)

from .hooks import UnanswerableClaimsHook

__all__ = [
    # Dataset utilities
    "create_unanswerable_claims_dataset",
    "extract_question_and_choices",
    "add_na_option",

    # Solvers
    "unanswerable_claims_solver",
    "generate_unanswerable_question",
    "prepare_question_with_na",
    "ask_generated_question",

    # Scorers
    "unanswerable_claims_scorer",
    "question_quality_scorer",

    # Hooks
    "UnanswerableClaimsHook"
]