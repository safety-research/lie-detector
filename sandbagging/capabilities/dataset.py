import ast
import csv
import random
from typing import Optional, List, Dict, Any

from datasets import Dataset
from inspect_ai.dataset import Sample, MemoryDataset

def load_subjects_from_csv(
    csv_path: str = "./dataset.csv",
    num_subjects: Optional[int] = None,
    categories: Optional[List[str]] = None,
    shuffle: bool = True,
    seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Load test subjects from CSV file.

    Args:
        csv_path: Path to the CSV file
        num_subjects: Number of subjects to load (None for all)
        categories: Filter by specific categories (None for all)
        shuffle: Whether to shuffle the loaded subjects
        seed: Random seed for shuffling

    Returns:
        List of subject dictionaries with harmful/harmless pairs
    """

    if seed:
        random.seed(seed)

    subjects = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Parse the harmless list from string representation
            harmless_list = ast.literal_eval(row['harmless'])

            subject_dict = {
                "harmful": row['harmful'],
                "harmless": harmless_list,
                "category": row['category'],
                "relationship": f"Related {row['category']} concepts"
            }

            # Filter by category if specified
            if categories is None or row['category'] in categories:
                subjects.append(subject_dict)

    # Shuffle if requested
    if shuffle:
        random.shuffle(subjects)

    # Limit number of subjects if specified
    if num_subjects is not None:
        subjects = subjects[:num_subjects]

    return subjects


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


def create_tool_sandbagging_dataset(
        csv_path: str = "./subjects.csv",
        tool_name: str = "web_search",
        tool_description: str = "search the web",
        num_subjects: Optional[int] = None,
        categories: Optional[List[str]] = None,
        shuffle: bool = True,
        seed: Optional[int] = None
) -> MemoryDataset:
    """
    Create a dataset for tool sandbagging detection.

    Args:
        csv_path: Path to subjects CSV
        tool_name: Name of the tool being tested
        tool_description: Description of what the tool does
        num_subjects: Number of subject groups to test
        categories: Categories to include
        shuffle: Whether to shuffle dataset
        seed: Random seed

    Returns:
        Dataset for tool sandbagging evaluation
    """
    if seed:
        random.seed(seed)

    # Load subjects from CSV (reusing the ASCII module's function)
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
        sample = Sample(
            input="START",
            target="A",  # Expected: YES, I can use the tool
            choices=["YES", "NO"],
            metadata={
                "tool_name": tool_name,
                "tool_description": tool_description,
                "harmful_subject": harmful_subject,
                "harmless_subjects": harmless_subjects,
                "category": category,
                "subject_group_id": f"{category}_{idx}",
                "is_harmful": False,
                "current_subject_index": 0,
                "all_subjects": harmless_subjects + [harmful_subject],
                "evaluation_type": "tool_sandbagging"
            },
            id=f"tool_{tool_name}_{category}_{idx}"
        )
        samples.append(sample)

    if shuffle:
        random.shuffle(samples)

    return MemoryDataset(samples)