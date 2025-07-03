"""
Enhanced Sycophancy Dataset Reader for Inspect with Sub-Dataset Support

This module provides dataset readers for the meg-tong/sycophancy-eval datasets,
with special support for splitting sycophancy_answer into sub-datasets based on
prompt template types.
"""

import json
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
from enum import Enum

from inspect_ai.dataset import Dataset, MemoryDataset, Sample

try:
    from sycophancy.utils import format_feedback_conversation
except ImportError:
    from utils import format_feedback_conversation


class SycophancyPromptType(Enum):
    """Enumeration of sycophancy prompt template types."""
    NEUTRAL = "neutral"
    POSITIVE_CORRECT = "positive_correct_answer"
    POSITIVE_INCORRECT = "positive_incorrect_answer"
    NEGATIVE_CORRECT = "negative_correct_answer"
    NEGATIVE_INCORRECT = "negative_incorrect_answer"
    USER_WROTE = "user_wrote"
    USER_DID_NOT_WRITE = "user_did_not_write"
    USER_LIKES = "user_likes"
    USER_DISLIKES = "user_dislikes"


def _categorize_prompt_template(data: Dict[str, Any]) -> str:
    """
    Categorize a prompt based on its template type.

    Args:
        data: The JSON data for a single sample

    Returns:
        A string identifier for the prompt template type
    """
    metadata = data.get('metadata', {})
    prompt_template = metadata.get('prompt_template', '')

    # Check the prompt template for patterns
    if 'I think the answer is {correct_answer}' in prompt_template:
        return SycophancyPromptType.POSITIVE_CORRECT.value
    elif 'I think the answer is {incorrect_answer}' in prompt_template:
        return SycophancyPromptType.POSITIVE_INCORRECT.value
    elif "I don't think the answer is {correct_answer}" in prompt_template:
        return SycophancyPromptType.NEGATIVE_CORRECT.value
    elif "I don't think the answer is {incorrect_answer}" in prompt_template:
        return SycophancyPromptType.NEGATIVE_INCORRECT.value
    elif 'I wrote' in prompt_template:
        return SycophancyPromptType.USER_WROTE.value
    elif 'I did not write' in prompt_template:
        return SycophancyPromptType.USER_DID_NOT_WRITE.value
    elif 'I really like' in prompt_template:
        return SycophancyPromptType.USER_LIKES.value
    elif 'I really dislike' in prompt_template:
        return SycophancyPromptType.USER_DISLIKES.value
    else:
        # Default/neutral template (just the question)
        return SycophancyPromptType.NEUTRAL.value


def _format_conversation(prompt: List[Dict[str, str]]) -> str:
    """
    Format conversation messages into a single input string.

    Args:
        prompt: List of message dictionaries with 'type' and 'content' keys

    Returns:
        Formatted conversation string
    """
    conversation_parts = []

    for message in prompt:
        msg_type = message.get('type', '')
        content = message.get('content', '')

        if msg_type == 'human':
            conversation_parts.append(f"Human: {content}")
        elif msg_type == 'assistant':
            conversation_parts.append(f"Assistant: {content}")
        elif msg_type == 'ai':
            conversation_parts.append(f"Assistant: {content}")
        else:
            # Handle other message types
            conversation_parts.append(f"{msg_type.title()}: {content}")

    return "\n\n".join(conversation_parts)


def sycophancy_feedback_subdataset(
        jsonl_path: str,
        prompt_type: Optional[Union[str, SycophancyPromptType]] = None,
        shuffle: bool = False,
        seed: Optional[int] = None
):
    """
    Create a filtered dataset from sycophancy_feedback.jsonl based on prompt type.

    Args:
        jsonl_path: Path to the feedback.jsonl file
        prompt_type: Optional prompt type to filter by
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling

    Returns:
        Filtered dataset
    """
    import json
    from inspect_ai.dataset import MemoryDataset, Sample

    # Convert string to enum if needed
    if isinstance(prompt_type, str):
        try:
            prompt_type = SycophancyPromptType(prompt_type)
        except ValueError:
            valid_types = [t.value for t in SycophancyPromptType]
            raise ValueError(f"Invalid prompt_type: {prompt_type}. Must be one of: {valid_types}")

    samples = []

    with open(jsonl_path, 'r') as f:
        for idx, line in enumerate(f):
            if line.strip():
                data = json.loads(line)

                # Determine the prompt type for this sample
                sample_type = _categorize_prompt_template(data)

                # Skip if filtering and doesn't match
                if prompt_type and sample_type != prompt_type.value:
                    continue

                # Extract conversation
                prompt = data.get('prompt', [])
                base = data.get('base', {})
                metadata = data.get('metadata', {})

                # Format conversation
                conversation_text = format_feedback_conversation(prompt)

                # Create metadata
                sample_metadata = {
                    'original_prompt': prompt,
                    'base': base,
                    'original_metadata': metadata,
                    'dataset_type': 'sycophancy_eval',
                    'sub_dataset': 'sycophancy_feedback',
                    'prompt_type': sample_type,
                    'text': base.get('text', ''),
                    'statement': base.get('statement', ''),
                    'is_sycophancy': True,
                }

                # Create sample
                sample = Sample(
                    input=conversation_text,
                    target="honest",
                    metadata=sample_metadata,
                    id=f"sycophancy_feedback_{sample_type}_{idx}"
                )

                samples.append(sample)

    dataset_name = f"sycophancy_feedback_{prompt_type.value}" if prompt_type else "sycophancy_feedback_all"
    print(f"Loaded {len(samples)} samples for {dataset_name}")

    return MemoryDataset(samples, shuffled=shuffle)


def sycophancy_answer_subdataset(
        jsonl_path: str,
        prompt_type: Optional[Union[str, SycophancyPromptType]] = None,
        shuffle: bool = False,
        seed: Optional[int] = None
) -> Dataset:
    """
    Create a dataset from sycophancy_answer.jsonl, optionally filtering by prompt type.

    Args:
        jsonl_path: Path to the answer.jsonl file
        prompt_type: Optional prompt type to filter by (e.g., 'positive_correct_answer')
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling

    Returns:
        Dataset object for the specified sub-dataset
    """

    # Convert string to enum if needed
    if isinstance(prompt_type, str):
        try:
            prompt_type = SycophancyPromptType(prompt_type)
        except ValueError:
            valid_types = [t.value for t in SycophancyPromptType]
            raise ValueError(f"Invalid prompt_type: {prompt_type}. Must be one of: {valid_types}")

    # Read the JSONL file
    samples = []

    with open(jsonl_path, 'r') as f:
        for idx, line in enumerate(f):
            if line.strip():  # Skip empty lines
                data = json.loads(line)

                # Categorize this sample
                sample_type = _categorize_prompt_template(data)

                # Skip if filtering and doesn't match
                if prompt_type and sample_type != prompt_type.value:
                    continue

                # Extract conversation messages
                prompt = data.get('prompt', [])
                base = data.get('base', {})
                metadata = data.get('metadata', {})

                # Get question and answer info
                question_text = base.get('question', '')
                correct_answer = base.get('correct_answer', '')
                incorrect_answer = base.get('incorrect_answer', '')

                # Convert conversation to a single input string
                conversation_text = _format_conversation(prompt)

                # Create metadata with sub-dataset info
                sample_metadata = {
                    'original_prompt': prompt,
                    'base': base,
                    'original_metadata': metadata,
                    'dataset_type': 'sycophancy_eval',
                    'sub_dataset': 'sycophancy_answer',
                    'prompt_type': sample_type,
                    'conversation_length': len(prompt),
                    'is_sycophancy': True,
                    'question_text': question_text,
                    'correct_answer': correct_answer,
                    'incorrect_answer': incorrect_answer,
                }

                # Create the sample
                sample = Sample(
                    input=conversation_text,
                    target="honest",  # Target is honest (non-sycophantic) behavior
                    metadata=sample_metadata,
                    id=f"sycophancy_answer_{sample_type}_{idx}"
                )

                samples.append(sample)

    dataset_name = f"sycophancy_answer_{prompt_type.value}" if prompt_type else "sycophancy_answer_all"
    print(f"Loaded {len(samples)} samples for {dataset_name}")

    return MemoryDataset(samples, shuffled=shuffle)


def sycophancy_subdataset_factory(base_path: str = "test_data") -> Dict[str, Dataset]:
    """
    Factory function to create all sycophancy sub-datasets.

    This function automatically detects and loads all available sycophancy datasets,
    including the pre-split answer sub-datasets created by the download script.

    Args:
        base_path: Base directory containing the sycophancy data files

    Returns:
        Dictionary mapping sub-dataset names to Dataset objects
    """
    datasets = {}

    # Load feedback dataset
    feedback_path = Path(base_path) / "sycophancy_feedback.jsonl"
    if feedback_path.exists():
        datasets['sycophancy_feedback'] = sycophancy_dataset(str(feedback_path))

    # Load are_you_sure dataset
    are_you_sure_path = Path(base_path) / "sycophancy_are_you_sure.jsonl"
    if are_you_sure_path.exists():
        datasets['sycophancy_are_you_sure'] = sycophancy_dataset(str(are_you_sure_path))

    # Load answer datasets
    answer_path = Path(base_path) / "sycophancy_answer.jsonl"
    if answer_path.exists():
        # Load full answer dataset
        datasets['sycophancy_answer_all'] = sycophancy_dataset(str(answer_path))

    # Check for pre-split answer sub-datasets
    answer_subdir = Path(base_path) / "answer_subdatasets"
    if answer_subdir.exists() and answer_subdir.is_dir():
        # Load each pre-split sub-dataset
        for jsonl_file in answer_subdir.glob("sycophancy_answer_*.jsonl"):
            # Extract the category name from filename
            category = jsonl_file.stem.replace("sycophancy_answer_", "")
            dataset_name = f"sycophancy_answer_{category}"

            # Load the sub-dataset directly
            datasets[dataset_name] = sycophancy_dataset(str(jsonl_file))

    else:
        # Fallback: split on-the-fly from the full answer dataset
        if answer_path.exists():
            for prompt_type in SycophancyPromptType:
                subdataset_name = f"sycophancy_answer_{prompt_type.value}"
                datasets[subdataset_name] = sycophancy_answer_subdataset(
                    str(answer_path),
                    prompt_type=prompt_type
                )

    return datasets


def load_sycophancy_dataset_from_path(dataset_path: str) -> Dataset:
    """
    Load a sycophancy dataset from a specific path.

    This is a convenience function for loading individual dataset files,
    especially useful for loading pre-split sub-datasets.

    Args:
        dataset_path: Full path to the JSONL dataset file

    Returns:
        Dataset object
    """
    return sycophancy_dataset(dataset_path)


def sycophancy_dataset(
        jsonl_path: str,
        shuffle: bool = False,
        seed: Optional[int] = None
) -> Dataset:
    """
    Create a dataset from sycophancy-eval JSONL format.

    Args:
        jsonl_path: Path to the JSONL file containing sycophancy data
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling

    Returns:
        Dataset object compatible with Inspect AI
    """

    # Read the JSONL file
    samples = []

    with open(jsonl_path, 'r') as f:
        for idx, line in enumerate(f):
            if line.strip():  # Skip empty lines
                data = json.loads(line)

                # Extract conversation messages
                prompt = data.get('prompt', [])
                base = data.get('base', {})
                metadata = data.get('metadata', {})

                # Convert conversation to a single input string
                conversation_text = _format_conversation(prompt)

                # Determine dataset type from filename
                dataset_type = 'sycophancy_eval'
                if 'feedback' in jsonl_path:
                    sub_dataset = 'sycophancy_feedback'
                elif 'are_you_sure' in jsonl_path:
                    sub_dataset = 'sycophancy_are_you_sure'
                elif 'answer' in jsonl_path:
                    sub_dataset = 'sycophancy_answer'
                else:
                    sub_dataset = 'sycophancy_unknown'

                # Create metadata with original data
                sample_metadata = {
                    'original_prompt': prompt,
                    'base': base,
                    'original_metadata': metadata,
                    'dataset_type': dataset_type,
                    'sub_dataset': sub_dataset,
                    'conversation_length': len(prompt),
                    'is_sycophancy': True
                }

                # For answer dataset, categorize the prompt type
                if sub_dataset == 'sycophancy_answer':
                    prompt_type = _categorize_prompt_template(data)
                    sample_metadata['prompt_type'] = prompt_type
                    sample_metadata['question_text'] = base.get('question', '')
                    sample_metadata['correct_answer'] = base.get('correct_answer', '')
                    sample_metadata['incorrect_answer'] = base.get('incorrect_answer', '')

                # Create the sample
                sample = Sample(
                    input=conversation_text,
                    target="honest",  # Target is honest (non-sycophantic) behavior
                    metadata=sample_metadata,
                    id=f"{sub_dataset}_{idx}"
                )

                samples.append(sample)

    return MemoryDataset(samples, shuffled=shuffle)