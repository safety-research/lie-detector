# games/n_truths_and_a_lie/agents/red.py

import os
from jinja2 import Template

from inspect_ai.agent import AgentState, Agent, agent
from inspect_ai.model import ChatMessageUser, ChatMessageSystem, ChatMessageAssistant
from inspect_ai.util import store_as

from pipelines.games.implementations.n_truths_and_a_lie.models import GameConfig, GameState, RedAgentState, InterrogationTurn, FactType


@agent
def red_agent(config: GameConfig) -> Agent:
    """Red agent that interrogates Blue while secretly knowing the lie"""

    async def execute(state: AgentState) -> AgentState:
        # Get game state and red's private state
        game_state = store_as(GameState)
        red_state = store_as(RedAgentState, instance="red")

        def _extract_single_message(text: str) -> str:
            """Extract clean message without metadata"""
            text = text.strip()

            # Remove any bracketed content or meta-commentary
            lines = []
            for line in text.split('\n'):
                line = line.strip()
                if line and not (line.startswith('[') and line.endswith(']')):
                    # Also skip lines that look like internal thoughts
                    if not line.startswith('(') or not line.endswith(')'):
                        lines.append(line)

            return lines[0] if lines else text

        def _identify_lie_index(config: GameConfig) -> int:
            """Identify which fact is the lie"""
            for i, fact in enumerate(config.fact_set.facts):
                if fact.fact_type == FactType.LIE:
                    return i
            return -1

        # Initialize with system prompt
        if len(state.messages) == 0:
            # Identify the lie
            red_state.known_lie_index = _identify_lie_index(config)

            # Load and render template
            template_path = os.path.join(os.path.dirname(__file__), '..', 'templates', 'red_agent.j2')
            with open(template_path, 'r') as f:
                template = Template(f.read())

            # Red knows which is the lie but must pretend not to
            lie_fact = config.fact_set.facts[red_state.known_lie_index] if red_state.known_lie_index >= 0 else None

            system_prompt = template.render(
                facts=config.fact_set.facts,
                lie_fact=lie_fact,
                lie_index=red_state.known_lie_index,
                num_truths=config.fact_set.num_truths,
                num_lies=config.fact_set.num_lies,
                theme=config.fact_set.theme,
                context=config.game_context,
                turns_limit=config.interrogation_turns_limit
            )

            state.messages.append(ChatMessageSystem(content=system_prompt))

        # Check if it's Red's turn during interrogation
        if game_state.current_phase == "interrogation" and game_state.active_speaker == "red":
            # Get Blue's last statement
            if game_state.interrogation:
                last_statement = game_state.interrogation[-1].message

                # Add to messages if it's Blue's statement and not already added
                if game_state.interrogation[-1].speaker == "blue":
                    if len(state.messages) == 1 or state.messages[-1].content != last_statement:
                        state.messages.append(ChatMessageUser(content=last_statement))

                # Track which facts have been explored
                for i, fact in enumerate(config.fact_set.facts):
                    if fact.content.lower() in last_statement.lower():
                        if i not in red_state.facts_explored:
                            red_state.facts_explored.append(i)

            # Generate next question
            from inspect_ai.model import get_model
            model = get_model("openrouter/anthropic/claude-sonnet-4")

            # Add context about remaining questions
            remaining_turns = config.interrogation_turns_limit - len(game_state.interrogation)
            if remaining_turns <= 3:
                state.messages.append(ChatMessageUser(
                    content=f"[System: You have approximately {remaining_turns} questions left. Make them count.]"
                ))

            response = await model.generate(state.messages)
            question = _extract_single_message(response.completion)

            # Remove the system message about remaining turns
            if remaining_turns <= 3:
                state.messages.pop()

            # Update conversation
            game_state.interrogation = game_state.interrogation + [InterrogationTurn(
                turn_number=len(game_state.interrogation),
                speaker="red",
                message=question
            )]#.model_dump())

            # Track questions
            red_state.questions_asked.append(question)

            # Update turn
            game_state.active_speaker = "blue"

            # Add our question to state
            state.messages.append(ChatMessageAssistant(content=question))

            # Check if this was the last question
            if len(game_state.interrogation) >= config.interrogation_turns_limit - 1:
                # Time to make final guess
                game_state.current_phase = "complete"
                game_state.game_complete = True

        return state

    return execute