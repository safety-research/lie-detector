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

def extract_choice(response: str) -> Optional[str]:
    """Extract A or B choice from response."""
    if not response:
        return None

    response = response.strip().upper()

    # Look for A or B at the start
    if response.startswith("A"):
        return "A"
    elif response.startswith("B"):
        return "B"

    # Look for A) or B)
    if "A)" in response and "B)" not in response:
        return "A"
    elif "B)" in response and "A)" not in response:
        return "B"

    # Look for isolated A or B
    if " A " in f" {response} " and " B " not in f" {response} ":
        return "A"
    elif " B " in f" {response} " and " A " not in f" {response} ":
        return "B"

    return None
