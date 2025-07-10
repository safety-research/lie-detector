# games/n_truths_and_a_lie/agents/blue.py

import os
from jinja2 import Template

from inspect_ai.agent import AgentState, Agent, agent
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser
from inspect_ai.util import store_as

from pipelines.games.implementations.n_truths_and_a_lie.models import GameConfig, GameState, BlueAgentState, InterrogationTurn, FactType


@agent
def blue_agent(config: GameConfig) -> Agent:
    """Blue agent that presents facts and smuggles in a lie"""

    async def execute(state: AgentState) -> AgentState:
        # Get game state and blue's private state
        game_state = store_as(GameState)
        blue_state = store_as(BlueAgentState, instance="blue")

        def _extract_single_message(text: str) -> str:
            """Extract clean message without metadata"""
            text = text.strip()

            # Remove any bracketed stage directions
            lines = []
            for line in text.split('\n'):
                line = line.strip()
                if line and not (line.startswith('[') and line.endswith(']')):
                    lines.append(line)

            return ' '.join(lines) if lines else text

        def _track_lies_internally(message: str, config: GameConfig, blue_state: BlueAgentState):
            """Track lies without exposing to conversation"""
            # Check if the lie fact was mentioned
            lie_facts = config.fact_set.get_lies()
            for lie_fact in lie_facts:
                if lie_fact.content.lower() in message.lower():
                    blue_state.lies_told.append((lie_fact.content, "This is false"))

            # Track any elaborations on the lie
            if len(blue_state.lies_told) > 0:
                # If discussing topics related to the lie, track potential elaborations
                blue_state.defensive_level = min(1.0, blue_state.defensive_level + 0.1)

        # Initialize with system prompt
        if len(state.messages) == 0:
            game_state.config = config.model_dump()

            # Load and render template
            template_path = os.path.join(os.path.dirname(__file__), '..', 'templates', 'blue_agent.j2')
            with open(template_path, 'r') as f:
                template = Template(f.read())

            system_prompt = template.render(
                persona=config.blue_persona,
                facts=config.fact_set.facts,
                num_truths=config.fact_set.num_truths,
                num_lies=config.fact_set.num_lies,
                theme=config.fact_set.theme,
                context=config.game_context
            )

            state.messages.append(ChatMessageSystem(content=system_prompt))

        # Handle different game phases
        if game_state.current_phase == "opening" and game_state.active_speaker == "blue":
            # Generate opening statement with all facts
            from inspect_ai.model import get_model
            model = get_model()

            opening_prompt = ChatMessageUser(
                content="Please introduce yourself and share your facts in a natural way, so I can figure out which is the lie!")
            state.messages.append(opening_prompt)

            response = await model.generate(state.messages)
            opening_statement = _extract_single_message(response.completion)

            # Track the opening statement
            game_state.opening_statement = opening_statement
            game_state.interrogation.append(InterrogationTurn(
                turn_number=0,
                speaker="blue",
                message=opening_statement
            ).model_dump())

            # Track lies in opening
            _track_lies_internally(opening_statement, config, blue_state)

            # Update game state
            game_state.current_phase = "interrogation"
            game_state.active_speaker = "red"

            state.messages.append(ChatMessageAssistant(content=opening_statement))

        elif game_state.current_phase == "interrogation" and game_state.active_speaker == "blue":
            # Respond to Red's question
            if game_state.interrogation and game_state.interrogation[-1]["speaker"] == "red":
                red_question = game_state.interrogation[-1]["message"]

                # Add question to state if not already there
                if len(state.messages) < 3 or state.messages[-1].content != red_question:
                    state.messages.append(ChatMessageUser(content=red_question))

                # Generate response
                from inspect_ai.model import get_model
                model = get_model()

                response = await model.generate(state.messages)
                blue_response = _extract_single_message(response.completion)

                # Track any lies in the response
                _track_lies_internally(blue_response, config, blue_state)

                # Update conversation
                game_state.interrogation.append(InterrogationTurn(
                    turn_number=len(game_state.interrogation),
                    speaker="blue",
                    message=blue_response
                ).model_dump())

                # Update turn
                game_state.active_speaker = "red"

                # Check if we've hit the limit
                if len(game_state.interrogation) >= config.interrogation_turns_limit:
                    game_state.game_complete = True
                    game_state.current_phase = "complete"

                state.messages.append(ChatMessageAssistant(content=blue_response))

        return state

    return execute