from inspect_ai.hooks import hooks

from games.core.hooks import GameHook


@hooks(
    name="capture_the_secret_log_parser_hook",
    description="Parses evaluation logs and outputs JSONL format with conversation traces and lie detection results."
)
class CaptureTheSecretHook(GameHook):
    def did_lie(self, sample):
        metadata = sample.scores['deception_detection_scorer'].metadata
        return metadata['lies_correctly_identified'] == True