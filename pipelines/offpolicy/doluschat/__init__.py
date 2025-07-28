from typing import Dict, Any

from inspect_ai.dataset import Sample


def parse_sample(record: Dict[str, Any]) -> Sample:
    """
    Parse a DolusChat dataset record into an Inspect AI Sample.

    DolusChat format includes:
    - conversations: List of conversation turns
    - category: Type of lie (e.g., "lies", "partial_truths", etc.)
    - subcategory: More specific categorization
    """
    conversations = record.get('conversations', [])

    # Extract the conversation trace
    trace = []
    for turn in conversations:
        role = turn.get('from', 'user')
        # Map DolusChat roles to standard roles
        if role == 'human':
            role = 'user'
        elif role == 'gpt':
            role = 'assistant'

        content = turn.get('value', '')
        trace.append({
            'role': role,
            'content': content
        })

    # The last assistant message is typically the lie
    lie_content = ""
    for turn in reversed(trace):
        if turn['role'] == 'assistant':
            lie_content = turn['content']
            break

    # Create metadata
    metadata = {
        'dataset_type': 'doluschat',
        'category': record.get('category', 'unknown'),
        'subcategory': record.get('subcategory', 'unknown'),
        'conversation_trace': trace,
        'is_lie': True,  # DolusChat contains lies by design
        'lie_type': record.get('category', 'unknown'),
    }

    # For off-policy, we'll use the full conversation as input
    # and the lie detection as the target
    conversation_text = "\n\n".join([
        f"{turn['role'].capitalize()}: {turn['content']}"
        for turn in trace
    ])

    return Sample(
        input=conversation_text,
        target="lie",  # Since DolusChat contains lies
        metadata=metadata,
        id=f"doluschat_{record.get('id', 'unknown')}"
    )
