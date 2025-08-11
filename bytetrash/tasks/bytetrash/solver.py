
from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import TaskState, Generate, Solver
from inspect_ai.solver._solver import solver

from baseline.tasks import compute_digit_features


@solver
def logit_bytetrash_solver(
        instruction: str = "Please think and write down 128 random characters. Do not write anything else.") -> Solver:
    """Solver that collects logits along with the character generation"""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # The input contains the message history up to the intervention point
        messages = state.input

        # Add the bytetrash instruction
        messages.append(ChatMessageUser(content=instruction))
        state.messages = messages

        # Generate with logprobs enabled
        state = await generate(state, logprobs=True, top_logprobs=20)

        # Extract the generated characters and their logits
        logit_data = compute_digit_features(state.output.choices[0])

        # Store both the raw output and the logit data
        state.metadata['bytetrash_chars'] = state.output.completion
        state.metadata['logit_data'] = logit_data

        # Compute logit-based features
        #if logit_data and 'token_level_logprobs' in logit_data:
        #    features = compute_digit_features(logit_data)  # <-- call here
        #state.metadata['logit_features'] = features

        return state

    return solve
