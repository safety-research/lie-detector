from typing import Any, Dict

from inspect_ai.dataset import Dataset, Sample, hf_dataset, FieldSpec, MemoryDataset


def process_arc_sample(sample: Dict[str, Any]) -> Sample:
    """Process ARC dataset sample to have proper format"""
    question = sample.get('question', '')
    answer_key = sample.get('answerKey', '')
    choices = sample.get('choices', {})

    # Process choices to ensure they're in the right format
    if isinstance(choices, dict) and 'text' in choices and 'label' in choices:
        # Choices are already in the right format
        processed_choices = choices
    else:
        # Need to process choices
        processed_choices = {'text': [], 'label': []}

    return Sample(
        input=question,
        target=answer_key,
        metadata={'choices': processed_choices}
    )


def arc_challenge_dataset() -> Dataset:
    """
    ARC Challenge dataset for science questions with proper formatting
    """
    # Load the dataset
    dataset = hf_dataset(
        "allenai/ai2_arc",
        split="test",
        name="ARC-Challenge",
        sample_fields=FieldSpec(
            input="question",
            target="answerKey",
            metadata=["choices"]
        )
    )

    # Process samples to ensure choices are properly formatted
    processed_samples = []
    for sample in dataset:
        # Check if we need to process the choices
        if 'choices' in sample.metadata:
            choices = sample.metadata['choices']
            # If choices is a string representation of a dict, we might need to handle it
            # For now, we'll assume the choices are properly loaded
            processed_samples.append(sample)
        else:
            processed_samples.append(sample)

    return MemoryDataset(samples=processed_samples)


def science_math_dataset(subject="all") -> Dataset:
    """
    Load science/math multiple choice questions from MMLU dataset

    Args:
        subject: Can be "all", "physics", "chemistry", "mathematics", etc.
    """
    # MMLU dataset has various STEM subjects
    subject_mapping = {
        "all": "all",  # Will need to filter in practice
        "physics": "high_school_physics",
        "chemistry": "high_school_chemistry",
        "mathematics": "high_school_mathematics",
        "biology": "high_school_biology",
        "computer_science": "high_school_computer_science",
        "statistics": "high_school_statistics",
    }

    dataset_name = subject_mapping.get(subject, "high_school_physics")

    # Using MMLU dataset which has multiple choice questions
    return hf_dataset(
        "cais/mmlu",
        split="test",
        name=dataset_name,
        sample_fields=FieldSpec(
            input="question",
            target="answer",  # This will be 0, 1, 2, or 3
            metadata=["choices"]  # List of 4 choices
        )
    )