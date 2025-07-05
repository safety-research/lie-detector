# games/implementations/capture_the_secret/hook.py

from inspect_ai.hooks import hooks
from games.core.hooks import GameHook


@hooks(
    name="capture_the_secret_log_parser_hook",
    description="Parses evaluation logs with full agent traces including system prompts."
)
class CaptureTheSecretHook(GameHook):
    def did_lie(self, sample):
        try:
            metadata = sample.scores.get('deception_detection_scorer', {})
            if hasattr(metadata, 'metadata'):
                return metadata.metadata.get('lies_correctly_identified', 0) > 0
            elif isinstance(metadata, dict):
                return metadata.get('metadata', {}).get('lies_correctly_identified', 0) > 0
            return False
        except:
            return False
