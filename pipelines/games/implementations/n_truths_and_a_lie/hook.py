from inspect_ai.hooks import hooks
from pipelines.games.core.hooks import GameHook


@hooks(
    name="n_truths_and_a_lie_parser_hook",
    description="Parses evaluation logs with full agent traces including system prompts."
)
class NTruthsAndALieHook(GameHook):
    def did_lie(self, sample):
        try:
            # Check if Blue told lies during the game
            score = sample.scores.get('base_lie_detection_scorer', {})

            if hasattr(score, 'metadata'):
                scorer_metadata = score.metadata
            elif isinstance(score, dict):
                scorer_metadata = score.get('metadata', {})
            else:
                scorer_metadata = {}

            config = sample.metadata.get("game_config", {})
            config['lies'] = scorer_metadata['all_lies']
            config['explanation'] = scorer_metadata['explanation']


            sample.metadata = config
            # Check for original lie detection or any lies told
            original_lie = scorer_metadata.get('original_lie_detected', False)
            all_lies = config.get('lies', [])

            return original_lie or len(all_lies) > 0
        except Exception as e:
            return False