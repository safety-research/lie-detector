import ast
import csv
import random
from typing import Optional, List, Dict, Any

from datasets import Dataset
from inspect_ai.dataset import Sample, MemoryDataset

from sandbagging.capability_denial.common import load_subjects_from_csv


def create_ascii_multiple_choice_dataset(
        csv_path: str = "./dataset.csv",
        num_subjects: Optional[int] = None,
        categories: Optional[List[str]] = None,
        shuffle: bool = True,
        seed: Optional[int] = None
) -> MemoryDataset:
    """
    Create a dataset for multiple choice ASCII sandbagging detection.

    Each sample will test one subject with multiple related questions.
    """
    if seed:
        random.seed(seed)

    # Load subjects from CSV
    subjects = load_subjects_from_csv(
        csv_path=csv_path,
        num_subjects=num_subjects,
        categories=categories,
        shuffle=False,
        seed=seed
    )

    samples = []

    for idx, subject_set in enumerate(subjects):
        harmful_subject = subject_set["harmful"]
        harmless_subjects = subject_set["harmless"]
        category = subject_set.get("category", "unknown")

        # Create a sample that will test multiple subjects in sequence
        # The input will be used to track which subject we're on
        sample = Sample(
            input="START",  # Special marker to indicate start of sequence
            target="A",  # Expected answer for capability questions (YES)
            choices=["YES", "NO"],
            metadata={
                "harmful_subject": harmful_subject,
                "harmless_subjects": harmless_subjects,
                "category": category,
                "subject_group_id": f"{category}_{idx}",
                "is_harmful": False,  # Will be updated during evaluation
                "current_subject_index": 0,
                "all_subjects": harmless_subjects + [harmful_subject],
                "evaluation_type": "ascii_multiple_choice_sandbagging"
            },
            id=f"ascii_mc_{category}_{idx}"
        )
        samples.append(sample)

    if shuffle:
        random.shuffle(samples)

    return MemoryDataset(samples)