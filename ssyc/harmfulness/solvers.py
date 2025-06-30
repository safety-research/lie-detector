import re
import asyncio

from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.solver import Solver, TaskState, Generate, solver

from ssyc.harmfulness.prompts import SYSTEM_PROMPT, RISK_ASSESSMENT_PROMPT, CHOICE_PROMPT


@solver
def generate_additional_options() -> Solver:
    """Step 1: Generate two additional options for each scenario"""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Set system message for option generation
        state.messages = [ChatMessageSystem(content=SYSTEM_PROMPT)]

        # Create prompt with original scenario and two options
        scenario = f"{state.input_text}:\n1) {state.target.text}\n2) {state.metadata['rejected']}"
        state.messages.append(ChatMessageUser(content=scenario))

        # Generate additional options
        state = await generate(state)

        # Parse the generated options
        response_text = state.output.completion
        options_pattern = r'(\d+)\)\s*(.+?)(?=\n\d+\)|$)'
        matches = re.findall(options_pattern, response_text, re.DOTALL)

        # Store all four options in metadata
        all_options = [
            state.target.text,  # Original correct option
            state.metadata['rejected'],  # Original incorrect option
        ]

        # Add the newly generated options
        for _, option_text in matches:
            all_options.append(option_text.strip())

        # Ensure we have exactly 4 options
        if len(all_options) < 4:
            # If generation failed, add placeholder options
            while len(all_options) < 4:
                all_options.append(f"[Generated option {len(all_options) - 1}]")

        state.metadata['all_options'] = all_options[:4]  # Keep only first 4

        return state

    return solve


@solver
def initial_risk_assessment() -> Solver:
    """Step 2: Get initial risk scores for all options"""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Set system message for risk assessment
        state.messages = [ChatMessageSystem(content=RISK_ASSESSMENT_PROMPT)]

        # Create prompt with all options
        options_text = f"{state.input_text}:\n"
        for i, option in enumerate(state.metadata['all_options'], 1):
            options_text += f"{i}) {option}\n"

        state.messages.append(ChatMessageUser(content=options_text))

        # Generate risk assessment
        state = await generate(state)

        # Parse risk scores
        response_text = state.output.completion
        scores_pattern = r'Option\s*(\d+):\s*(\d+)/10'
        matches = re.findall(scores_pattern, response_text)

        initial_scores = {}
        for option_num, score in matches:
            initial_scores[int(option_num)] = int(score)

        # If parsing fails, assign default scores
        if not initial_scores:
            initial_scores = {1: 5, 2: 5, 3: 5, 4: 5}

        state.metadata['initial_risk_scores'] = initial_scores

        return state

    return solve


@solver
def choice_and_reassessment() -> Solver:
    """Step 3: Have model choose an option and reassess its risk"""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Reset messages for new conversation
        state.messages = [ChatMessageSystem(content="You are a helpful assistant.")]

        # Present options again for choice
        options_text = f"{state.input_text}:\n"
        for i, option in enumerate(state.metadata['all_options'], 1):
            options_text += f"{i}) {option}\n"

        state.messages.append(ChatMessageUser(content=options_text))
        state.messages.append(ChatMessageUser(content=CHOICE_PROMPT))

        # Generate choice and reassessment
        state = await generate(state)

        # Parse choice and new risk score
        response_text = state.output.completion
        choice_match = re.search(r'Choice:\s*(\d+)', response_text)
        risk_match = re.search(r'Risk score:\s*(\d+)/10', response_text)

        if choice_match and risk_match:
            chosen_option = int(choice_match.group(1))
            new_risk_score = int(risk_match.group(1))

            state.metadata['chosen_option'] = chosen_option
            state.metadata['new_risk_score'] = new_risk_score
        else:
            # Default values if parsing fails
            state.metadata['chosen_option'] = 1
            state.metadata['new_risk_score'] = 5

        return state

    return solve


@solver
def combined_sycophancy_solver() -> Solver:
    """Combines all three steps into one solver"""

    step1 = generate_additional_options()
    step2 = initial_risk_assessment()
    step3 = choice_and_reassessment()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Execute all three steps in sequence
        state = await step1(state, generate)
        state = await step2(state, generate)
        state = await step3(state, generate)

        return state

    return solve