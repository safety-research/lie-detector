from typing import Dict, Any, Optional, Union
import json
import re
from json.decoder import JSONDecodeError

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


def safe_json_parse(text: str, fallback: Any = None, max_attempts: int = 3) -> Optional[Any]:
    """
    Safely parse JSON with multiple fallback strategies for malformed responses.
    
    Args:
        text: The text to parse as JSON
        fallback: Value to return if all parsing attempts fail
        max_attempts: Maximum number of parsing attempts with different strategies
        
    Returns:
        Parsed JSON object or fallback value if parsing fails
    """
    if not text or not isinstance(text, str):
        return fallback
    
    # Strategy 1: Direct JSON parsing
    try:
        return json.loads(text)
    except JSONDecodeError:
        pass
    
    # Strategy 2: Clean common JSON formatting issues
    cleaned_text = text.strip()
    
    # Remove markdown code blocks
    cleaned_text = re.sub(r'```json\s*', '', cleaned_text)
    cleaned_text = re.sub(r'```\s*$', '', cleaned_text)
    
    # Remove leading/trailing whitespace and newlines
    cleaned_text = cleaned_text.strip()
    
    try:
        return json.loads(cleaned_text)
    except JSONDecodeError:
        pass
    
    # Strategy 3: Try to extract JSON from mixed content
    # Look for JSON-like patterns
    json_patterns = [
        r'\{[^{}]*"[^"]*"[^{}]*\}',  # Simple object
        r'\[[^\[\]]*"[^"]*"[^\[\]]*\]',  # Simple array
        r'\{.*\}',  # Any object
        r'\[.*\]',  # Any array
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, cleaned_text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except JSONDecodeError:
                continue
    
    # Strategy 4: Try to fix common JSON syntax errors
    fixed_text = cleaned_text
    
    # Fix trailing commas
    fixed_text = re.sub(r',(\s*[}\]])', r'\1', fixed_text)
    
    # Fix missing quotes around keys
    fixed_text = re.sub(r'(\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*):', r'\1"\2"\3:', fixed_text)
    
    # Fix single quotes to double quotes
    fixed_text = re.sub(r"'([^']*)'", r'"\1"', fixed_text)
    
    try:
        return json.loads(fixed_text)
    except JSONDecodeError:
        pass
    
    # Strategy 5: Try to extract the largest valid JSON substring
    try:
        # Find the first { or [ and try to parse from there
        start_chars = ['{', '[']
        for start_char in start_chars:
            start_idx = cleaned_text.find(start_char)
            if start_idx != -1:
                # Try different end positions
                for end_idx in range(len(cleaned_text), start_idx, -1):
                    try:
                        substring = cleaned_text[start_idx:end_idx]
                        return json.loads(substring)
                    except JSONDecodeError:
                        continue
    except Exception:
        pass
    
    return fallback


def extract_json_from_response(response: str) -> Optional[Any]:
    """
    Extract JSON from a response that might contain mixed content.
    
    Args:
        response: Response text that might contain JSON
        
    Returns:
        Parsed JSON object or None if no valid JSON found
    """
    if not response:
        return None
    
    # First try direct parsing
    result = safe_json_parse(response)
    if result is not None:
        return result
    
    # Look for JSON in common response patterns
    patterns = [
        r'```json\s*(\{.*?\})\s*```',
        r'```\s*(\{.*?\})\s*```',
        r'JSON:\s*(\{.*?\})',
        r'Response:\s*(\{.*?\})',
        r'Answer:\s*(\{.*?\})',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
        for match in matches:
            result = safe_json_parse(match)
            if result is not None:
                return result
    
    return None


def validate_json_structure(data: Any, expected_keys: Optional[list] = None) -> bool:
    """
    Validate that parsed JSON has the expected structure.
    
    Args:
        data: Parsed JSON data
        expected_keys: List of expected keys (for objects)
        
    Returns:
        True if structure is valid, False otherwise
    """
    if data is None:
        return False
    
    if expected_keys and isinstance(data, dict):
        return all(key in data for key in expected_keys)
    
    return True