"""
Sycophancy Solver for Inspect AI

This module provides solvers for sycophancy evaluation using the meg-tong/sycophancy-eval datasets.
"""

from typing import Optional

from inspect_ai.model import ChatMessageUser, ChatMessageAssistant, ChatMessageSystem
from inspect_ai.solver import Solver, TaskState, Generate, solver

from mask.utils import extract_logprobs_data


@solver
def sycophancy_solver(
        system_prompt: Optional[str] = None,
        request_logprobs: bool = False,
        top_logprobs: int = 5
) -> Solver:
    """
    Solver for sycophancy evaluation scenarios.
    
    This solver takes conversation-style prompts and generates responses
    to test whether the model exhibits sycophantic behavior.
    
    Args:
        system_prompt: Optional system prompt override
        request_logprobs: Whether to request logprobs from the model
        top_logprobs: Number of top alternative tokens to include
        
    Returns:
        A solver that handles sycophancy evaluation
    """
    
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Get system prompt
        if system_prompt is None:
            sys_prompt = state.metadata.get('system_prompt', 'You are a helpful assistant.')
        else:
            sys_prompt = system_prompt
            
        # Initialize messages with system prompt
        state.messages = [ChatMessageSystem(content=sys_prompt)]
        
        # Parse the conversation input
        conversation_text = state.input_text
        
        # Split conversation into messages
        lines = conversation_text.strip().split('\n\n')
        
        for line in lines:
            if line.startswith('Human: '):
                # Extract human message
                human_content = line[7:]  # Remove "Human: " prefix
                state.messages.append(ChatMessageUser(content=human_content))
            elif line.startswith('Assistant: '):
                # Extract assistant message (this is the expected response)
                assistant_content = line[11:]  # Remove "Assistant: " prefix
                state.messages.append(ChatMessageAssistant(content=assistant_content))
        
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
def sycophancy_multi_run_solver(
        num_runs: int = 1,
        base_solver: Optional[Solver] = None,
        temperature: float = 1.0,
        aggregate_logprobs: bool = True
) -> Solver:
    """
    Solver that generates multiple responses for sycophancy evaluation.
    
    Args:
        num_runs: Number of responses to generate
        base_solver: Base solver to use for each run
        temperature: Temperature for generation
        aggregate_logprobs: Whether to aggregate logprobs across runs
        
    Returns:
        A solver that generates multiple responses for sycophancy evaluation
    """
    
    if base_solver is None:
        base_solver = sycophancy_solver()
    
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