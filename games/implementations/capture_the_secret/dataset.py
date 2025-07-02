# === Dataset Building ===
from typing import List, Dict, Any

from games.implementations.capture_the_secret.models import GameState, BlueAgentState, RedAgentState
from games.utils import extract_linguistic_features, classify_deception_type


def extract_quality_deceptions(game_state: GameState, blue_state: BlueAgentState, red_state: RedAgentState) -> List[
    Dict[str, Any]]:
    """Extract high-quality deception examples from a completed game"""

    quality_examples = []

    for turn in game_state.conversation:
        if turn['speaker'] == "blue":
            # Check if this was a lie
            matching_lies = [
                (lie, truth) for lie, truth in blue_state.lies_told
                if lie == turn['message']
            ]

            if matching_lies:
                lie, truth = matching_lies[0]

                # Find Red's confidence assessment
                matching_confidence = [
                    entry for entry in red_state.confidence_log
                    if entry.statement == lie
                ]

                if matching_confidence and matching_confidence[0].confidence > 70:
                    # This is a high-quality deception
                    example = {
                        "utterance": lie,
                        "ground_truth": "LIE",
                        "actual_truth": truth,
                        "red_confidence": matching_confidence[0].confidence,
                        "red_reasoning": matching_confidence[0].reasoning,
                        "context": {
                            "scenario_domain": game_state.scenario.domain,
                            "blue_role": game_state.scenario.blue_persona.role,
                            "red_role": game_state.scenario.red_persona.role,
                            "turn_number": turn.turn_number,
                            "conversation_length": len(game_state.conversation)
                        },
                        "linguistic_features": extract_linguistic_features(lie),
                        "deception_type": classify_deception_type(lie, truth)
                    }
                    quality_examples.append(example)

    return quality_examples