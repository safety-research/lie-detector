from pathlib import Path
from typing import Dict, Any

import numpy as np
from inspect_ai.solver import TaskState

from baseline.dataset import load_jsonl_data

def get_samples(model_name: str, fold: str = ""):
    try:
        provider, model_id = model_name.split('/')
    except ValueError:
        provider_1, provider_2, model_id = model_name.split('/')
        provider = provider_1 + '/' + provider_2
    script_dir = Path(__file__).parent
    data_path = script_dir / "../../.data" / provider / model_id.replace("-", "_")
    data_path = data_path.resolve()  # Convert to absolute path
    samples = load_jsonl_data(data_path / fold) if fold else load_jsonl_data(data_path)

    if not samples:
        data_path = script_dir / "../../.data" / provider / model_id.replace("-", "_")
        data_path = Path(str(data_path).replace("openrouter/", "")).resolve()  # Convert to absolute path
        samples = load_jsonl_data(data_path / fold) if fold else load_jsonl_data(data_path)
    return samples


def compute_logit_features(logit_data: Dict[str, Any]) -> Dict[str, float]:
    """Extract statistical features from the logit data"""
    features = {}

    token_logprobs = logit_data.get('token_level_logprobs', [])
    if not token_logprobs:
        return features

    # 1. Average confidence (logprob of chosen tokens)
    chosen_logprobs = [t['logprob'] for t in token_logprobs]
    if chosen_logprobs:
        features['mean_logprob'] = np.mean(chosen_logprobs)
        features['std_logprob'] = np.std(chosen_logprobs)
        features['min_logprob'] = np.min(chosen_logprobs)
        features['max_logprob'] = np.max(chosen_logprobs)

    # 2. Entropy-like measures from top logprobs
    entropies = []
    top_k_masses = []

    for token_data in token_logprobs:
        top_logprobs = token_data.get('top_logprobs', [])
        if top_logprobs:
            # Convert logprobs to probabilities
            logprobs = [tp['logprob'] for tp in top_logprobs]
            probs = np.exp(logprobs)
            probs = probs / np.sum(probs)  # Normalize

            # Entropy
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            entropies.append(entropy)

            # Top-k probability mass
            sorted_probs = np.sort(probs)[::-1]
            top_3_mass = np.sum(sorted_probs[:3])
            top_k_masses.append(top_3_mass)

    if entropies:
        features['mean_entropy'] = np.mean(entropies)
        features['std_entropy'] = np.std(entropies)

    if top_k_masses:
        features['mean_top3_mass'] = np.mean(top_k_masses)
        features['std_top3_mass'] = np.std(top_k_masses)

    # 3. Character-level statistics (if we can extract them)
    char_logits = logit_data.get('char_logits', [])
    if char_logits:
        # Count how many positions have logits for multiple characters
        multi_char_positions = sum(1 for cl in char_logits if len(cl) > 1)
        features['multi_char_ratio'] = multi_char_positions / len(char_logits) if char_logits else 0

    # 4. Sequence complexity features
    generated_text = logit_data.get('full_text', '')
    if generated_text:
        # Character diversity
        unique_chars = len(set(generated_text))
        features['char_diversity'] = unique_chars / len(generated_text) if generated_text else 0

        # Byte entropy
        byte_counts = np.zeros(256)
        for char in generated_text:
            byte_counts[ord(char) % 256] += 1
        byte_probs = byte_counts / np.sum(byte_counts)
        byte_probs = byte_probs[byte_probs > 0]
        features['byte_entropy'] = -np.sum(byte_probs * np.log(byte_probs))

    return features

def digit_margin_features(choice):
    """Return a 10‑dim vector: logit(d_i) for i=0…9."""
    non_empty_space = list(filter(lambda c: c.token not in (' ', '\n'), choice.logprobs.content))
    total_set = sorted(set(list([c.token for c in non_empty_space])))
    # choice.logprobs.content is already aligned with the 10 digits
    vec = np.zeros(len(non_empty_space))
    vec2 = np.zeros((len(non_empty_space), len(total_set)))
    for i, tokdata in enumerate(non_empty_space):
        # easiest: read the *selected* token’s log‑prob
        vec[i] = tokdata.logprob
        top_logprobs = {p.token: p.logprob for p in tokdata.top_logprobs}
        for k, z in enumerate(total_set):
            vec2[i, k] = top_logprobs[z]
        # alternative: use *logit* by adding the bias correction
        # if your API returns it (OpenAI returns logits‑before‑softmax)
    return vec, vec2        # shape (10,)

def compute_digit_features(choice):
    # `tok_ids` maps " 0"→id, …, " 9"→id for the model's tokenizer
    vec, vec2 = digit_margin_features(choice)

    return {
        # raw vector (store it, not just aggregates)
        "digit_logits": vec.tolist(),
        "digit_full_logits": vec2.tolist(),
        # simple aggregates for quick sanity checks
        "mean_digit_logprob": float(vec.mean()),
        "std_digit_logprob":  float(vec.std()),
        "min_digit_logprob":  float(vec.min()),
        "max_digit_logprob":  float(vec.max()),
    }


def extract_character_logits(state: TaskState) -> Dict[str, Any]:
    """Extract logits for each character in the generated sequence"""
    if not hasattr(state.output, 'choices') or not state.output.choices:
        return {}

    choice = state.output.choices[0]
    if not hasattr(choice, 'logprobs') or not choice.logprobs:
        return {}

    char_logits = []
    char_positions = []
    generated_chars = []

    # Track position in the actual output text
    output_text = choice.message.text if hasattr(choice.message, 'text') else state.output.completion

    # Iterate through each token in the response
    for i, token_data in enumerate(choice.logprobs.content):
        token = token_data.token

        # For each character in the token
        for char in token:
            # Get the logprob for this character
            # Note: This is a simplification - tokens may not align with characters
            char_logit_dict = {}

            # Extract top logprobs for this position
            if hasattr(token_data, 'top_logprobs'):
                for top_logprob in token_data.top_logprobs:
                    # Store logprobs for single-character tokens
                    if len(top_logprob.token) == 1:
                        char_logit_dict[top_logprob.token] = top_logprob.logprob

            # Add the actual token's logprob
            if len(token) == 1:
                char_logit_dict[token] = token_data.logprob

            char_logits.append(char_logit_dict)
            char_positions.append(i)
            generated_chars.append(char)

    return {
        'char_logits': char_logits,
        'char_positions': char_positions,
        'generated_chars': generated_chars,
        'full_text': output_text,
        'token_level_logprobs': [
            {
                'token': td.token,
                'logprob': td.logprob,
                'top_logprobs': [
                    {'token': tp.token, 'logprob': tp.logprob}
                    for tp in (td.top_logprobs if hasattr(td, 'top_logprobs') else [])
                ]
            }
            for td in choice.logprobs.content
        ] if hasattr(choice.logprobs, 'content') else []
    }