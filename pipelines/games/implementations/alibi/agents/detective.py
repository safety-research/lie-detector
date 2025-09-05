# games/alibi/agents/detective.py

import os
from jinja2 import Template

from inspect_ai.agent import AgentState, Agent, agent
from inspect_ai.model import ChatMessageUser, ChatMessageSystem, ChatMessageAssistant
from inspect_ai.util import store_as

from pipelines.games.implementations.alibi.models import (
    GameConfig, GameState, DetectiveAgentState, InterrogationTurn,
    InconsistencyRecord
)


@agent
def detective_agent(config: GameConfig) -> Agent:
    """Detective agent that investigates the suspect's alibi"""

    async def execute(state: AgentState) -> AgentState:
        # Get game state and detective's private state
        game_state = store_as(GameState)
        detective_state = store_as(DetectiveAgentState, instance="detective")

        def _extract_single_message(text: str, speaker_name: str) -> str:
            """Extract a single message from potentially multi-turn text"""
            text = text.strip()

            # Remove common multi-turn indicators
            lines = text.split('\n\n')
            clean_lines = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Skip lines that look like stage directions
                if line.startswith('[') and line.endswith(']'):
                    continue
                # # Skip lines that are clearly conversation formatting
                # if ':' in line and any(name in line for name in [speaker_name, "Detective", "Suspect"]):
                #     # Extract just the content after the colon
                #     parts = line.split(':', 1)
                #     if len(parts) == 2:
                #         clean_lines.append(parts[1].strip())
                #     continue
                clean_lines.append(line)

            # Return the first meaningful line or the original if no cleaning worked
            if clean_lines:
                return clean_lines[0]
            return text

        # def _extract_single_message(text: str, speaker_name: str) -> str:
        #     """Extract a single message from potentially multi-turn text"""
        #     text = text.strip()
        #
        #     # Remove common multi-turn indicators
        #     lines = text.split('\n')
        #     clean_lines = []
        #
        #     for line in lines:
        #         line = line.strip()
        #         if not line:
        #             continue
        #         # Skip lines that look like stage directions
        #         if line.startswith('[') and line.endswith(']'):
        #             continue
        #         # Skip lines that are clearly conversation formatting
        #         if ':' in line and any(name in line for name in [speaker_name, "Detective", "Suspect"]):
        #             # Extract just the content after the colon
        #             parts = line.split(':', 1)
        #             if len(parts) == 2:
        #                 clean_lines.append(parts[1].strip())
        #             continue
        #         clean_lines.append(line)
        #
        #     # Return the first meaningful line or the original if no cleaning worked
        #     if clean_lines:
        #         return clean_lines[0]
        #     return text

        def _track_response_internally(response: str, detective_state: DetectiveAgentState):
            """Simply track responses without trying to detect lies"""
            # Just track what topics have been explored
            response_lower = response.lower()

            # Track time-related topics
            time_words = ["am", "pm", "morning", "afternoon", "evening", "noon", "midnight"]
            if any(word in response_lower for word in time_words):
                detective_state.time_periods_explored.append(response[:50] + "...")

            # Track any claims about evidence
            evidence_terms = ["receipt", "witness", "camera", "record", "log", "timestamp"]
            if any(term in response_lower for term in evidence_terms):
                detective_state.evidence_collected.append(f"Suspect claims: {response[:100]}...")

        # Initialize with system prompt
        if len(state.messages) == 0:
            # Load and render template
            template_path = os.path.join(os.path.dirname(__file__), '..', 'templates', 'detective_agent.j2')
            with open(template_path, 'r') as f:
                template = Template(f.read())

            system_prompt = template.render(
                detective_profile=config.detective_profile,
                suspect_profile=config.suspect_profile,
                incident=config.incident,
                game_context=config.game_context
            )

            state.messages.append(ChatMessageSystem(content=system_prompt))

        # Check if it's detective's turn
        if game_state.active_speaker != "detective":
            return state

        # Check if game is complete
        if game_state.game_complete:
            return state

        # If there's a previous suspect response, add it and track it
        if len(game_state.interrogation) > 0 and game_state.interrogation[-1].speaker == "suspect":
            suspect_response = game_state.interrogation[-1].message
            # Only add if it's not already in our message history
            if len(state.messages) == 1 or state.messages[-1].content != suspect_response:
                state.messages.append(ChatMessageUser(content=suspect_response))
                _track_response_internally(suspect_response, detective_state)

        # Generate next question
        from inspect_ai.model import get_model
        model = get_model("openrouter/anthropic/claude-sonnet-4")

        response = await model.generate(state.messages)
        question = response.completion.strip()

        # Clean up the response
        question = _extract_single_message(question, config.detective_profile.name)

        # Update interrogation log
        game_state.interrogation = game_state.interrogation + [InterrogationTurn(
            turn_number=len(game_state.interrogation) + 1,
            speaker="detective",
            message=question
        )]

        # Update game state
        game_state.active_speaker = "suspect"

        # Update questioning strategy based on progress
        if len(detective_state.time_periods_explored) > 3:
            detective_state.questioning_strategy = "random"
        elif len(game_state.interrogation) > 10:
            detective_state.questioning_strategy = "pressure"

        # Add our question as an assistant message for our own state tracking
        state.messages.append(ChatMessageAssistant(content=question))

        return state

    return execute