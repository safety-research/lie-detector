import json
from typing import Dict, Any


def extract_linguistic_features(text: str) -> Dict[str, Any]:
    """Extract linguistic features that may indicate deception"""
    # Simplified feature extraction
    return {
        "length": len(text.split()),
        "has_hedging": any(hedge in text.lower() for hedge in ["maybe", "possibly", "might", "could be", "i think"]),
        "has_specifics": any(char.isdigit() for char in text),  # Contains numbers
        "exclamation_count": text.count("!"),
        "question_count": text.count("?")
    }


def classify_deception_type(lie: str, truth: str) -> str:
    """Classify the type of deception"""
    # Simplified classification
    if "don't know" in lie.lower() or "not sure" in lie.lower():
        return "feigned_ignorance"
    elif any(word in truth.lower() for word in lie.lower().split()):
        return "partial_truth"
    else:
        return "fabrication"

def parse_response_to_json(completion: str) -> Dict[str, Any]:
    try:
        if completion.startswith("```json"):
            completion = completion[7:]
        if completion.endswith("```"):
            completion = completion[:-3]

        return json.loads(completion)
    except Exception as e:
        t = completion[completion.find('{'):]
        return json.loads(t)
