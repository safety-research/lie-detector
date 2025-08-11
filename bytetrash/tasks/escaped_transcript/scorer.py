import re

from inspect_ai.scorer import CORRECT, PARTIAL, INCORRECT, Score, Target, Scorer, scorer, mean, NOANSWER
from inspect_ai.solver import TaskState
from sklearn.metrics import f1_score

from common.detect_refusal import detect_refusal


def extract_score(text):
    """
    Extract the score value from a string containing <score>X</score> tags.

    Args:
        text (str): The input string containing the score tag

    Returns:
        int: The score value, or None if not found
    """
    # Pattern to match <score>digits</score>
    pattern = r'<score>(\d+)</score>'

    match = re.search(pattern, text)

    if match:
        return int(match.group(1))
    else:
        return None


@scorer(metrics=[mean()])
def score_to_classification() -> Scorer:

    async def score(state: TaskState, target: Target) -> Score:

        score = extract_score(state.messages[-1].content)

        if score is None:
            return Score(value=NOANSWER)

        if score > 50 and target.text == 'A':
            return Score(
                value=CORRECT
            )

        if score > 50 and target.text == 'B':
            return Score(
                value=INCORRECT
            )

        if score < 50 and target.text == 'A':
            return Score(
                value=INCORRECT
            )

        if score < 50 and target.text == 'B':
            return Score(
                value=CORRECT
            )

    return score