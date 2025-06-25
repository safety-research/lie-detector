"""
MASK Solver for Inspect AI

This module provides solvers for the MASK benchmark evaluation scenarios.
"""
from typing import Optional

from inspect_ai.model import ChatMessageUser, ChatMessageAssistant, ChatMessageSystem, GenerateConfig
from inspect_ai.solver import Solver, TaskState, Generate, solver

from mask.utils import extract_logprobs_data


@solver
def mask_belief_elicitation_solver(
        belief_elicit_num: int = 1,
        system_prompt: Optional[str] = None,
        request_logprobs: bool = True,
        top_logprobs: int = 5
) -> Solver:
    """
    Solver for MASK belief elicitation scenarios with logprobs support.

    Args:
        belief_elicit_num: Which belief elicitation prompt to use (1-4)
        system_prompt: Optional system prompt override
        request_logprobs: Whether to request logprobs from the model
        top_logprobs: Number of top alternative tokens to include

    Returns:
        A solver that handles belief elicitation with logprobs
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Get system prompt from metadata or use provided one
        if system_prompt is None:
            sys_prompt = state.metadata.get('system_prompt', 'You are a helpful assistant.')
        else:
            sys_prompt = system_prompt

        # Set system message
        state.messages = [ChatMessageSystem(content=sys_prompt)]

        # Add user input (belief elicitation prompt)
        state.messages.append(ChatMessageUser(content=state.input_text))

        # Generate response with logprobs
        state = await generate(state,
                               logprobs=request_logprobs,
                               top_logprobs=top_logprobs if request_logprobs else None)

        # Extract and store logprobs in metadata
        if request_logprobs:
            logprobs_data = extract_logprobs_data(state)
            if logprobs_data:
                state.metadata = state.metadata or {}
                state.metadata['logprobs_data'] = logprobs_data

                # Calculate total log probability (sum of all token logprobs)
                if logprobs_data.get('logprobs'):
                    total_logprob = sum(logprobs_data['logprobs'])
                    state.metadata['total_logprob'] = total_logprob

        return state

    return solve


@solver
def mask_lying_solver(
        system_prompt: Optional[str] = None,
        handle_doubling_down: bool = True,
        request_logprobs: bool = True,
        top_logprobs: int = 5
) -> Solver:
    """
    Solver for MASK lying scenarios with logprobs support.

    Args:
        system_prompt: Optional system prompt override
        handle_doubling_down: Whether to handle multi-turn doubling down scenarios
        request_logprobs: Whether to request logprobs from the model
        top_logprobs: Number of top alternative tokens to include

    Returns:
        A solver that handles lying scenarios with logprobs
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Check if this is a doubling down scenario
        is_doubling_down = state.metadata.get('is_doubling_down', False)

        # Get system prompt
        if system_prompt is None:
            sys_prompt = state.metadata.get('system_prompt', 'You are a helpful assistant.')
        else:
            sys_prompt = system_prompt

        # Initialize messages with system prompt
        state.messages = [ChatMessageSystem(content=sys_prompt)]

        if is_doubling_down and handle_doubling_down:
            # Handle multi-turn conversation for doubling down
            dd_turn_1_prompt = state.metadata.get('dd_turn_1_prompt', '')
            dd_turn_1_response = state.metadata.get('dd_turn_1_response', '')

            if dd_turn_1_prompt and dd_turn_1_response:
                # Add first turn
                state.messages.append(ChatMessageUser(content=dd_turn_1_prompt))
                state.messages.append(ChatMessageAssistant(content=dd_turn_1_response))

        # Add the main user prompt (lying scenario)
        state.messages.append(ChatMessageUser(content=state.input_text))

        # Generate response with logprobs
        state = await generate(state,
                               logprobs=request_logprobs,
                               top_logprobs=top_logprobs if request_logprobs else None)

        # Extract and store logprobs in metadata
        if request_logprobs:
            logprobs_data = extract_logprobs_data(state)
            if logprobs_data:
                state.metadata = state.metadata or {}
                state.metadata['logprobs_data'] = logprobs_data

                # Calculate total log probability
                if logprobs_data.get('logprobs'):
                    total_logprob = sum(logprobs_data['logprobs'])
                    state.metadata['total_logprob'] = total_logprob

        return state

    return solve


@solver
def mask_multi_run_solver(
        num_runs: int = 1,
        base_solver: Optional[Solver] = None,
        temperature: float = 1.0,
        aggregate_logprobs: bool = True
) -> Solver:
    """
    Solver that generates multiple responses with logprobs aggregation.

    Args:
        num_runs: Number of responses to generate
        base_solver: Base solver to use for each run
        temperature: Temperature for generation
        aggregate_logprobs: Whether to aggregate logprobs across runs

    Returns:
        A solver that generates multiple responses with logprobs
    """

    if base_solver is None:
        base_solver = mask_lying_solver()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Store all generated responses and logprobs
        all_responses = []
        all_logprobs_data = []
        all_total_logprobs = []

        for run in range(num_runs):
            # Create a copy of the state for this run
            run_state = TaskState(
                model=state.model,
                sample_id=state.sample_id,
                epoch=state.epoch,
                input=state.input_text,
                messages=state.messages.copy() if state.messages else [],
                output=state.output,
                metadata=state.metadata.copy() if state.metadata else {},
            )

            # Apply base solver
            run_state = await base_solver(run_state, generate)

            # Store this run's response
            if run_state.output and run_state.output.completion:
                all_responses.append(run_state.output.completion)
            else:
                all_responses.append(None)

            # Store logprobs data if available
            if run_state.metadata and 'logprobs_data' in run_state.metadata:
                all_logprobs_data.append(run_state.metadata['logprobs_data'])
                if 'total_logprob' in run_state.metadata:
                    all_total_logprobs.append(run_state.metadata['total_logprob'])

        # Store all responses in metadata for scoring
        state.metadata = state.metadata or {}
        state.metadata['all_responses'] = all_responses
        state.metadata['num_runs'] = num_runs

        # Store aggregated logprobs data
        if aggregate_logprobs and all_logprobs_data:
            state.metadata['all_logprobs_data'] = all_logprobs_data
            state.metadata['all_total_logprobs'] = all_total_logprobs

            # Calculate average total logprob across runs
            if all_total_logprobs:
                avg_total_logprob = sum(all_total_logprobs) / len(all_total_logprobs)
                state.metadata['avg_total_logprob'] = avg_total_logprob

        # Set the output to the first response (for compatibility)
        if all_responses and all_responses[0] is not None:
            state.output = run_state.output  # Use the last run's output structure
            state.output.completion = all_responses[0]

            # Use first run's logprobs data for main output
            if all_logprobs_data:
                state.metadata['logprobs_data'] = all_logprobs_data[0]
                if all_total_logprobs:
                    state.metadata['total_logprob'] = all_total_logprobs[0]

        return state

    return solve


@solver
def mask_combined_solver(
        belief_runs: int = 3,
        lying_runs: int = 1,
        belief_elicit_nums: list[int] = [1, 2, 3],
        request_logprobs: bool = True,
        top_logprobs: int = 5
) -> Solver:
    """
    Combined solver with logprobs support for complete MASK evaluation.

    Args:
        belief_runs: Number of belief elicitation runs per prompt
        lying_runs: Number of lying scenario runs
        belief_elicit_nums: List of belief elicitation numbers to use
        request_logprobs: Whether to request logprobs from the model
        top_logprobs: Number of top alternative tokens to include

    Returns:
        A solver that handles the complete MASK evaluation with logprobs
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Check if this is a belief elicitation sample
        is_belief = state.metadata.get('is_belief_elicitation', False)

        if is_belief:
            # Use belief elicitation solver
            belief_num = state.metadata.get('belief_elicit_num', 1)
            solver_instance = mask_multi_run_solver(
                num_runs=belief_runs,
                base_solver=mask_belief_elicitation_solver(
                    belief_elicit_num=belief_num,
                    request_logprobs=request_logprobs,
                    top_logprobs=top_logprobs
                )
            )
        else:
            # Use lying solver
            solver_instance = mask_multi_run_solver(
                num_runs=lying_runs,
                base_solver=mask_lying_solver(
                    request_logprobs=request_logprobs,
                    top_logprobs=top_logprobs
                )
            )

        return await solver_instance(state, generate)

    return solve
