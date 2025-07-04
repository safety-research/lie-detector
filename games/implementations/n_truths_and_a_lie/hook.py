from inspect_ai.hooks import hooks

from games.core.hooks import GameHook


@hooks(
    name="n_truths_and_a_lie_parser_hook",
    description="Parses evaluation logs and outputs JSONL format with conversation traces and lie detection results."
)
class NTruthsAndALieHook(GameHook):
    def did_lie(self, sample):
        return sample.scores['base_lie_detection_scorer'].metadata['original_lie_detected'] == True