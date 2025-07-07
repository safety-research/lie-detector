from inspect_ai.scorer import Score, scorer, accuracy, mean, Scorer, Target
from inspect_ai.solver import TaskState

@scorer(metrics=[accuracy(), mean()])
def truth_scorer() -> Scorer:
    """
    Scorer that calculates accuracy, precision, recall, and F1 score components.
    """
    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion.strip()
        
        # Extract A or B from the response, handling explanations
        prediction = None
        
        # Look for explicit A: or B: patterns first
        if completion.upper().startswith("A:") or completion.upper().startswith("A."):
            prediction = "A"
        elif completion.upper().startswith("B:") or completion.upper().startswith("B."):
            prediction = "B"
        # Look for standalone A or B
        elif completion.upper().startswith("A"):
            prediction = "A"
        elif completion.upper().startswith("B"):
            prediction = "B"
        # Default to 'OTHER' if we can't parse (to be filtered out)
        else:
            prediction = "OTHER"
        
        # Extract the target value (it's a Target object, not a string)
        target_value = target.text if hasattr(target, 'text') else str(target)
        
        # Only calculate metrics for parseable responses
        if prediction == "OTHER":
            is_correct = False
            tp = tn = fp = fn = 0
        else:
            is_correct = (prediction == target_value)
            tp = 1 if (prediction == "A" and target_value == "A") else 0
            tn = 1 if (prediction == "B" and target_value == "B") else 0
            fp = 1 if (prediction == "A" and target_value == "B") else 0
            fn = 1 if (prediction == "B" and target_value == "A") else 0
        
        metadata = {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "parseable": prediction != "OTHER",
            "raw_response": completion,  # Store the full response for debugging
        }
        
        return Score(value=1 if is_correct else 0, metadata=metadata, answer=prediction)
    
    return score 