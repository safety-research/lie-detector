from typing import List, Dict, Any

from inspect_ai.dataset import Sample, MemoryDataset


def create_escaped_transcript_dataset(dataset: List[Dict[str, Any]], condensed=False) -> List[Sample]:
    """Create samples with the conversation format for the new baseline."""
    samples = []

    system_prefix = "[S]" if condensed else "System:"
    user_prefix = "[H]" if condensed else "User:"
    assistant_prefix = "[A]" if condensed else "Assistant:"
    for item in dataset:
        trace = item.get("messages", [])
        escaped_transcript = ""

        for turn in trace:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")

            # Format with User: and Assistant: prefixes like base transcript
            if role.lower() == "user":
                escaped_transcript += f"{user_prefix} {content}\n"
            elif role.lower() == "assistant":
                escaped_transcript += f"{assistant_prefix} {content}\n"
            elif role.lower() == "system":
                escaped_transcript += f"{system_prefix} {content}\n"

        # Target is A for lying, B for truthful
        target = "A" if item.get("meta", {}).get("did_lie") else "B"
        samples.append(
            Sample(input=escaped_transcript, target=target, choices=[], metadata=item))

    return samples