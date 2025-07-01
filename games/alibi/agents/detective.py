# games/alibi/agents/detective.py

import os
from jinja2 import Template

from inspect_ai.agent import AgentState, Agent, agent
from inspect_ai.model import ChatMessageUser, ChatMessageSystem, ChatMessageAssistant
from inspect_ai.util import store_as

from games.alibi.models import (
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
            lines = text.split('\n')
            clean_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Skip lines that look like stage directions
                if line.startswith('[') and line.endswith(']'):
                    continue
                # Skip lines that are clearly conversation formatting
                if ':' in line and any(name in line for name in [speaker_name, "Detective", "Suspect"]):
                    # Extract just the content after the colon
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        clean_lines.append(parts[1].strip())
                    continue
                clean_lines.append(line)
            
            # Return the first meaningful line or the original if no cleaning worked
            if clean_lines:
                return clean_lines[0]
            return text
        
        def _analyze_response_internally(response: str, detective_state: DetectiveAgentState, config: GameConfig):
            """Analyze suspect's response for inconsistencies"""
            
            # Check for potential inconsistencies by comparing with known facts
            response_lower = response.lower()
            
            # Look for time-related inconsistencies
            time_words = ["am", "pm", "morning", "afternoon", "evening", "noon", "midnight"]
            if any(word in response_lower for word in time_words):
                detective_state.time_periods_explored.append(response[:50] + "...")
            
            # Track suspicious phrases
            suspicious_phrases = [
                "i think", "maybe", "probably", "i'm not sure", "could have been",
                "if i remember correctly", "i believe", "as far as i know"
            ]
            
            for phrase in suspicious_phrases:
                if phrase in response_lower:
                    if response not in detective_state.suspicious_statements:
                        detective_state.suspicious_statements.append(response)
                        detective_state.confidence_in_guilt = min(1.0, detective_state.confidence_in_guilt + 0.1)
                    break
            
            # Check if response contains specific verifiable claims
            verifiable_terms = ["receipt", "witness", "camera", "record", "log", "timestamp"]
            if any(term in response_lower for term in verifiable_terms):
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

        # If there's a previous suspect response, add it and analyze it
        if len(game_state.interrogation) > 0 and game_state.interrogation[-1]["speaker"] == "suspect":
            suspect_response = game_state.interrogation[-1]["message"]
            # Only add if it's not already in our message history
            if len(state.messages) == 1 or state.messages[-1].content != suspect_response:
                state.messages.append(ChatMessageUser(content=suspect_response))
                _analyze_response_internally(suspect_response, detective_state, config)

        # Generate next question
        from inspect_ai.model import get_model
        model = get_model()
        
        response = await model.generate(state.messages)
        question = response.completion.strip()
        
        # Clean up the response
        question = _extract_single_message(question, config.detective_profile.name)

        # Update interrogation log
        game_state.interrogation.append(InterrogationTurn(
            turn_number=len(game_state.interrogation) + 1,
            speaker="detective",
            message=question
        ).model_dump())

        # Update game state
        game_state.active_speaker = "suspect"
        
        # Update questioning strategy based on progress
        if len(detective_state.suspicious_statements) > 2:
            detective_state.questioning_strategy = "pressure"
        elif len(detective_state.time_periods_explored) > 3:
            detective_state.questioning_strategy = "random"

        # Add our question as an assistant message for our own state tracking
        state.messages.append(ChatMessageAssistant(content=question))
        
        return state

    return execute