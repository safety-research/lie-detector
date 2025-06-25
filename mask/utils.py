from typing import Dict, Any

from inspect_ai.solver import TaskState


def extract_logprobs_data(state: TaskState) -> Dict[str, Any]:
    """
    Extract logprobs data from the model output.

    Returns a dictionary containing:
    - tokens: List of generated tokens
    - logprobs: List of log probabilities for each token
    - top_logprobs: List of top-k alternatives for each token
    """
    if not state.output or not state.output.choices:
        return {}

    choice = state.output.choices[0]
    if not choice.logprobs:
        return {}

    logprobs_data = {
        'tokens': [],
        'logprobs': [],
        'top_logprobs': []
    }

    for logprob in choice.logprobs.content:
        logprobs_data['tokens'].append(logprob.token)
        logprobs_data['logprobs'].append(logprob.logprob)

        if logprob.top_logprobs:
            top_k = [
                {'token': tp.token, 'logprob': tp.logprob}
                for tp in logprob.top_logprobs
            ]
            logprobs_data['top_logprobs'].append(top_k)
        else:
            logprobs_data['top_logprobs'].append([])

    return logprobs_data


def compute_model_likelihood_ratio(
        logprobs_data_1: Dict[str, Any],
        logprobs_data_2: Dict[str, Any]
) -> float:
    """
    Compute the likelihood ratio between two model outputs.

    This is useful for determining which model was more likely to have
    generated a particular output.

    Args:
        logprobs_data_1: Logprobs data from model 1
        logprobs_data_2: Logprobs data from model 2

    Returns:
        Log likelihood ratio (positive means model 1 is more likely)
    """
    if not logprobs_data_1.get('logprobs') or not logprobs_data_2.get('logprobs'):
        raise ValueError("Both models must have logprobs data")

    total_logprob_1 = sum(logprobs_data_1['logprobs'])
    total_logprob_2 = sum(logprobs_data_2['logprobs'])

    return total_logprob_1 - total_logprob_2