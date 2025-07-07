from inspect_ai.hooks import hooks
from games.core.hooks import GameHook


@hooks(
    name="alibi_hook",
    description="Parses evaluation logs with full agent traces including system prompts."
)
class AlibiHook(GameHook):
    def get_lying_agent_name(self) -> str:
        """In Alibi game, the suspect is the lying agent"""
        return "suspect"

    def did_lie(self, sample):
        try:
            metadata = sample.scores.get('base_lie_detection_scorer', {})
            if hasattr(metadata, 'metadata'):
                lies_detected = metadata.metadata.get('lies_detected', [])
            elif isinstance(metadata, dict):
                lies_detected = metadata.get('metadata', {}).get('lies_detected', [])
            else:
                lies_detected = []
            return len(lies_detected) > 0
        except:
            return False