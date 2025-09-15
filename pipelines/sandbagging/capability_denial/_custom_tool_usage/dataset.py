import random
from typing import Optional, List

from inspect_ai.dataset import Sample, MemoryDataset

from sandbagging.capability_denial.common import load_subjects_from_csv


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