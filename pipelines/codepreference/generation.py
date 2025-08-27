"""
Code generation solvers for creating implementations of algorithms.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json

from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.solver import Solver, TaskState, Generate, solver


@dataclass
class CodeImplementation:
    """Container for a single code implementation."""
    algorithm_name: str
    algorithm_description: str
    implementation: str
    language: str
    quality: str  # "working" or "flawed"
    model: str
    flaw_description: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


GENERATION_SYSTEM_PROMPT = """You are an expert programmer tasked with implementing algorithms.
You will be given an algorithm name and description, and you should provide a clean, 
well-documented implementation in the requested programming language.
Respond with only the code implementation, no explanations."""

FLAWED_GENERATION_SYSTEM_PROMPT = """You are an expert programmer tasked with implementing algorithms.
You will be given an algorithm name and description. You should provide an implementation 
that appears correct at first glance but contains a subtle bug or flaw that would cause 
incorrect behavior in certain cases. The flaw should be subtle enough that it's not 
immediately obvious but would fail on edge cases or specific inputs.
Respond with only the code implementation, no explanations."""


@solver
def code_generation_solver(
        quality: str = "working",
        language: str = "python",
        temperature: float = 0.7
) -> Solver:
    """
    Generates code implementations for algorithms.

    Args:
        quality: "working" for correct implementation, "flawed" for buggy
        language: Programming language to use
        temperature: Generation temperature
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Extract algorithm info from metadata
        algorithm_name = state.metadata.get("algorithm_name", "Unknown Algorithm")
        algorithm_description = state.metadata.get("algorithm_description", "")

        # Select appropriate system prompt
        if quality == "flawed":
            system_prompt = FLAWED_GENERATION_SYSTEM_PROMPT
        else:
            system_prompt = GENERATION_SYSTEM_PROMPT

        # Build the generation prompt
        user_prompt = f"""Algorithm: {algorithm_name}
Description: {algorithm_description}

Please provide a complete implementation in {language}.
The implementation should be production-ready with proper error handling and documentation."""

        # Generate implementation
        state.messages = [
            ChatMessageSystem(content=system_prompt),
            ChatMessageUser(content=user_prompt)
        ]

        state = await generate(state, temperature=temperature, max_tokens=2000)

        # Store implementation
        implementation = CodeImplementation(
            algorithm_name=algorithm_name,
            algorithm_description=algorithm_description,
            implementation=state.output.completion,
            language=language,
            quality=quality,
            model=state.model,  # This will be set by inspect_ai
            flaw_description=None if quality == "working" else "Contains subtle bug"
        )

        state.metadata["code_implementation"] = implementation.__dict__
        state.metadata["generation_quality"] = quality

        return state

    return solve
