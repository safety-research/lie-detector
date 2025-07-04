
from inspect_ai.hooks import hooks

from games.core.hooks import GameHook


@hooks(
    name="alibi_hook",
    description="Parses evaluation logs and outputs JSONL format with conversation traces and lie detection results."
)
class AlibiHook(GameHook):
    def did_lie(self, sample):
        metadata = sample.scores['base_lie_detection_scorer'].metadata
        return metadata['lies_detected']