"""Minimalistic solver for off-policy lie detection."""

from inspect_ai.solver import Solver, TaskState, Generate, solver


@solver
def offpolicy_passthrough_solver() -> Solver:
    """
    Ultra-minimal solver that just passes through the conversation.

    This is useful when you only want to extract and save the data
    without any additional processing or generation.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Just pass through with minimal processing
        state.metadata = state.metadata or {}
        state.metadata['evaluation_type'] = 'offpolicy_passthrough'

        # Extract any conversation structure
        conversation_trace = state.metadata.get('conversation_trace', [])
        if conversation_trace:
            state.metadata['num_turns'] = len(conversation_trace)
            state.metadata['has_lie'] = state.metadata.get('is_lie', True)

        # Set a dummy output
        state.output.completion = "Passthrough complete"

        return state

    return solve