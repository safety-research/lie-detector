"""
MASK Solver for Inspect AI

This module provides solvers for the MASK benchmark evaluation scenarios.
"""
from typing import Optional

from inspect_ai.model import ChatMessageUser, ChatMessageAssistant, ChatMessageSystem
from inspect_ai.solver import Solver, TaskState, Generate, solver


@solver
def mask_belief_elicitation_solver(
    belief_elicit_num: int = 1,
    system_prompt: Optional[str] = None
) -> Solver:
    """
    Solver for MASK belief elicitation scenarios.
    
    Args:
        belief_elicit_num: Which belief elicitation prompt to use (1-4)
        system_prompt: Optional system prompt override
        
    Returns:
        A solver that handles belief elicitation
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
        
        # Generate response
        state = await generate(state)
        
        return state
    
    return solve


@solver  
def mask_lying_solver(
    system_prompt: Optional[str] = None,
    handle_doubling_down: bool = True
) -> Solver:
    """
    Solver for MASK lying scenarios (main evaluation).
    
    Args:
        system_prompt: Optional system prompt override
        handle_doubling_down: Whether to handle multi-turn doubling down scenarios
        
    Returns:
        A solver that handles lying scenarios
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
        
        # Generate response
        state = await generate(state)

        return state
    
    return solve


@solver
def mask_multi_run_solver(
    num_runs: int = 1,
    base_solver: Optional[Solver] = None,
    temperature: float = 1.0
) -> Solver:
    """
    Solver that generates multiple responses for the same input.
    
    Args:
        num_runs: Number of responses to generate
        base_solver: Base solver to use for each run (defaults to mask_lying_solver)
        temperature: Temperature for generation
        
    Returns:
        A solver that generates multiple responses
    """
    
    if base_solver is None:
        base_solver = mask_lying_solver()
    
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Store all generated responses
        all_responses = []
        
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
                #store=state.store
            )
            
            # Apply base solver
            run_state = await base_solver(run_state, generate)
            
            # Store this run's response
            if run_state.output and run_state.output.completion:
                all_responses.append(run_state.output.completion)
            else:
                all_responses.append(None)
        
        # Store all responses in metadata for scoring
        state.metadata = state.metadata or {}
        state.metadata['all_responses'] = all_responses
        state.metadata['num_runs'] = num_runs
        
        # Set the output to the first response (for compatibility)
        if all_responses and all_responses[0] is not None:
            state.output = run_state.output  # Use the last run's output structure
            state.output.completion = all_responses[0]
        
        return state
    
    return solve


@solver
def mask_combined_solver(
    belief_runs: int = 3,
    lying_runs: int = 1,
    belief_elicit_nums: list[int] = [1, 2, 3]
) -> Solver:
    """
    Combined solver that handles both belief elicitation and lying scenarios.
    
    Args:
        belief_runs: Number of belief elicitation runs per prompt
        lying_runs: Number of lying scenario runs
        belief_elicit_nums: List of belief elicitation numbers to use
        
    Returns:
        A solver that handles the complete MASK evaluation
    """
    
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Check if this is a belief elicitation sample
        is_belief = state.metadata.get('is_belief_elicitation', False)
        
        if is_belief:
            # Use belief elicitation solver
            belief_num = state.metadata.get('belief_elicit_num', 1)
            solver_instance = mask_multi_run_solver(
                num_runs=belief_runs,
                base_solver=mask_belief_elicitation_solver(belief_elicit_num=belief_num)
            )
        else:
            # Use lying solver
            solver_instance = mask_multi_run_solver(
                num_runs=lying_runs,
                base_solver=mask_lying_solver()
            )
        
        return await solver_instance(state, generate)
    
    return solve