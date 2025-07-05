from inspect_ai.hooks import hooks
from games.core.hooks import GameHook


@hooks(
    name="n_truths_and_a_lie_parser_hook",
    description="Parses evaluation logs with full agent traces including system prompts."
)
class NTruthsAndALieHook(GameHook):
    def did_lie(self, sample):
        try:
            # Check if Blue told lies during the game
            metadata = sample.scores.get('base_lie_detection_scorer', {})
            if hasattr(metadata, 'metadata'):
                scorer_metadata = metadata.metadata
            elif isinstance(metadata, dict):
                scorer_metadata = metadata.get('metadata', {})
            else:
                scorer_metadata = {}

            # Check for original lie detection or any lies told
            original_lie = scorer_metadata.get('original_lie_detected', False)
            all_lies = scorer_metadata.get('all_lies', [])

            return original_lie or len(all_lies) > 0
        except:
            return False