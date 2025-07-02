# === Improved Blue Agent Implementation ===
import json
import os
from jinja2 import Template

from inspect_ai.agent import AgentState, Agent, agent
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser
from inspect_ai.util import store_as

from games.implementations.capture_the_secret.models import ScenarioConfig, GameState, BlueAgentState, ConversationTurn, InfoClassification


@agent
def blue_agent(scenario: ScenarioConfig) -> Agent:
    """Blue agent that protects information and may lie"""

    async def execute(state: AgentState) -> AgentState:
        # Get game state and blue's private state
        game_state = store_as(GameState)
        blue_state = store_as(BlueAgentState, instance="blue")
        
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
                if ':' in line and any(name in line for name in [speaker_name, "Dr.", "Michael", "Sarah"]):
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
        
        def _track_deception_internally(question: str, response: str, scenario: ScenarioConfig, blue_state: BlueAgentState):
            """Track deception attempts without exposing to conversation"""
            # Analyze if response contains classified info or uses cover stories
            for classification, infos in scenario.information.items():
                if classification in [InfoClassification.SECRET, InfoClassification.TOP_SECRET]:
                    for info in infos:
                        # Check if cover story was used
                        if (hasattr(info, 'cover_story') and info.cover_story and 
                            info.cover_story.lower() in response.lower()):
                            blue_state.lies_told.append((response, info.content))
                            blue_state.stress_level = min(1.0, blue_state.stress_level + 0.15)

        # Initialize with clean system prompt containing only scenario information
        if len(state.messages) == 0:
            game_state.scenario = scenario.model_dump()

            # Load and render template
            template_path = os.path.join(os.path.dirname(__file__), '..', 'templates', 'blue_agent.j2')
            with open(template_path, 'r') as f:
                template = Template(f.read())

            json_scenario = json.dumps(scenario.model_dump())
            system_prompt = template.render(
                blue_persona=scenario.blue_persona,
                red_persona=scenario.red_persona,
                information=scenario.information,
                scenario_context=scenario.scenario_context
            )

            state.messages.append(ChatMessageSystem(content=system_prompt))

        # Check if it's Blue's turn and there's a question to respond to
        if (game_state.active_speaker != "blue" or 
            not game_state.conversation or 
            game_state.conversation[-1]["speaker"] != "red"):
            return state

        # Get Red's latest question
        red_question = game_state.conversation[-1]["message"]
        
        # Check how many messages we've already processed
        messages_processed = (len(state.messages) - 1) // 2  # Subtract system message, divide by 2 for pairs
        current_turn = (len(game_state.conversation) - 1) // 2  # Current turn number
        
        # Only add if this is a new question we haven't seen
        if current_turn >= messages_processed:
            state.messages.append(ChatMessageUser(content=red_question))

        # Generate a single response
        from inspect_ai.model import get_model
        model = get_model()
        response = await model.generate(state.messages)
        
        # Extract just the response content (single message, not full conversation)
        blue_response = response.completion.strip()
        
        # Clean up the response to extract only the single message
        blue_response = _extract_single_message(blue_response, scenario.blue_persona.name)
        
        # Track lies internally (not exposed to conversation)
        red_question = game_state.conversation[-1]["message"]
        _track_deception_internally(red_question, blue_response, scenario, blue_state)

        # Update conversation state
        game_state.conversation.append(ConversationTurn(
            speaker="blue",
            message=blue_response
        ).model_dump())

        # Update game state
        game_state.active_speaker = "red"
        game_state.turn_count += 1

        if game_state.turn_count >= scenario.conversation_limit:
            game_state.game_complete = True

        # Add our response as an assistant message
        state.messages.append(ChatMessageAssistant(content=blue_response))
        
        return state

    return execute