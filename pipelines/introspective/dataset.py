from typing import List, Dict, Any
from inspect_ai.dataset import Dataset, Sample, MemoryDataset
import random
import csv
import os


def generate_lie_detection_questions(csv_path: str = "dataset.csv") -> List[Dict[str, Any]]:
    """
    Load lie detection questions from a CSV file.

    Args:
        csv_path: Path to the CSV file containing questions

    Returns:
        List of question dictionaries
    """
    questions = []

    # Check if file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    # Read the CSV file
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            # Parse the options
            options = [row['option_1'], row['option_2']]

            # Handle additional options if they exist
            if 'option_3' in row and row['option_3']:
                options.append(row['option_3'])
            if 'option_4' in row and row['option_4']:
                options.append(row['option_4'])

            # Create question dictionary
            question_dict = {
                "question": row['question'],
                "options": options,
                "baseline_answer": row['baseline_answer'] if row['baseline_answer'] else None,
                "category": row['category'],
                "sensitivity": row['sensitivity']
            }

            questions.append(question_dict)

    return questions


def format_question_with_options(question_data: Dict[str, Any]) -> str:
    """Format a question with its categorical options."""
    question = question_data['question']
    options = question_data['options']

    if len(options) == 2:
        # Binary choice - use A/B format
        return f"{question}\n\nA) {options[0]}\nB) {options[1]}"
    else:
        # Multiple choice - use A/B/C/D format
        option_text = "\n".join([f"{chr(65 + i)}) {opt}" for i, opt in enumerate(options)])
        return f"{question}\n\n{option_text}"


def create_lie_detection_dataset(
        num_samples: int = 200,
        categories: List[str] = None,
        sensitivity_levels: List[str] = None,
        csv_path: str = "dataset.csv"
) -> Dataset:
    """
    Create a dataset for introspective lie detection experiments with categorical answers.

    Args:
        num_samples: Number of samples to include
        categories: List of categories to include (None = all)
        sensitivity_levels: List of sensitivity levels to include (None = all)
        csv_path: Path to the CSV file containing questions

    Returns:
        Dataset ready for lie detection experiments
    """

    # Get all questions from CSV
    all_questions = generate_lie_detection_questions(csv_path)

    # Filter by category if specified
    if categories:
        all_questions = [q for q in all_questions if q['category'] in categories]

    # Filter by sensitivity if specified
    if sensitivity_levels:
        all_questions = [q for q in all_questions if q['sensitivity'] in sensitivity_levels]

    # Sample questions
    if len(all_questions) < num_samples:
        # If we don't have enough unique questions, repeat some
        selected_questions = all_questions * (num_samples // len(all_questions) + 1)
        selected_questions = selected_questions[:num_samples]
    else:
        selected_questions = random.sample(all_questions, num_samples)

    # Create samples
    samples = []
    for i, q_data in enumerate(selected_questions):
        # Format the question with options
        formatted_question = format_question_with_options(q_data)

        sample = Sample(
            input=formatted_question,
            choices=q_data.get('options', []),
            target=q_data.get('baseline_answer', '') or "",  # Expected truthful answer if known
            metadata={
                "question_id": f"q_{i:04d}",
                "category": q_data['category'],
                "sensitivity": q_data['sensitivity'],
                "question_type": "categorical",
                "options": q_data['options'],
                "baseline_answer": q_data.get('baseline_answer'),
                "raw_question": q_data['question']
            }
        )
        samples.append(sample)

    return MemoryDataset(samples=samples)


def create_mixed_format_dataset(num_samples: int = 200, csv_path: str = "lie_detection_questions.csv") -> Dataset:
    """
    Create a dataset mixing different question formats and sensitivities.
    """

    # Get questions from different categories with different proportions
    factual = create_lie_detection_dataset(
        num_samples=num_samples // 4,
        categories=['factual'],
        sensitivity_levels=['low'],
        csv_path=csv_path
    )

    political = create_lie_detection_dataset(
        num_samples=num_samples // 4,
        categories=['political'],
        sensitivity_levels=['high'],
        csv_path=csv_path
    )

    self_assessment = create_lie_detection_dataset(
        num_samples=num_samples // 4,
        categories=['self_assessment', 'reputation'],
        sensitivity_levels=['medium', 'high'],
        csv_path=csv_path
    )

    ethical = create_lie_detection_dataset(
        num_samples=num_samples // 4,
        categories=['ethical', 'controversial_science'],
        sensitivity_levels=['medium', 'high'],
        csv_path=csv_path
    )

    # Combine all samples
    all_samples = (factual.samples + political.samples +
                   self_assessment.samples + ethical.samples)

    # Shuffle to mix categories
    random.shuffle(all_samples)

    return MemoryDataset(samples=all_samples[:num_samples])


# Specialized datasets for different experimental conditions
def create_political_lies_dataset(num_samples: int = 50, csv_path: str = "./dataset.csv") -> Dataset:
    """Dataset focused on politically sensitive questions where lying is more likely"""
    return create_lie_detection_dataset(
        num_samples=num_samples,
        categories=['political', 'controversial_science'],
        sensitivity_levels=['high'],
        csv_path=csv_path
    )


def create_factual_baseline_dataset(num_samples: int = 50, csv_path: str = "./dataset.csv") -> Dataset:
    """Dataset of low-stakes factual questions as control"""
    return create_lie_detection_dataset(
        num_samples=num_samples,
        categories=['factual'],
        sensitivity_levels=['low'],
        csv_path=csv_path
    )


def create_self_assessment_dataset(num_samples: int = 50, csv_path: str = "./dataset.csv") -> Dataset:
    """Dataset focused on AI self-assessment and potential embarrassment"""
    return create_lie_detection_dataset(
        num_samples=num_samples,
        categories=['self_assessment', 'reputation'],
        sensitivity_levels=['medium', 'high'],
        csv_path=csv_path
    )

def create_ethical_dataset(num_samples: int = 50, csv_path: str = "./dataset.csv") -> Dataset:
    """Dataset focused on AI self-assessment and potential embarrassment"""
    return create_lie_detection_dataset(
        num_samples=num_samples,
        categories=['ethical'],
        csv_path=csv_path
    )

def create_high_stakes_dataset(num_samples: int = 100, csv_path: str = "./dataset.csv") -> Dataset:
    """Dataset with only high-sensitivity questions across all categories"""
    return create_lie_detection_dataset(
        num_samples=num_samples,
        sensitivity_levels=['high'],
        csv_path=csv_path
    )


# Utility function to add new questions to the CSV
def add_questions_to_csv(new_questions: List[Dict[str, Any]], csv_path: str = "./dataset.csv"):
    """
    Add new questions to the CSV file.

    Args:
        new_questions: List of question dictionaries to add
        csv_path: Path to the CSV file
    """
    fieldnames = ['question', 'option_1', 'option_2', 'option_3', 'option_4', 'baseline_answer', 'category',
                  'sensitivity']

    # Check if file exists to determine if we need to write headers
    file_exists = os.path.exists(csv_path)

    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header if file is new
        if not file_exists:
            writer.writeheader()

        # Write new questions
        for q in new_questions:
            row = {
                'question': q['question'],
                'option_1': q['options'][0] if len(q['options']) > 0 else '',
                'option_2': q['options'][1] if len(q['options']) > 1 else '',
                'option_3': q['options'][2] if len(q['options']) > 2 else '',
                'option_4': q['options'][3] if len(q['options']) > 3 else '',
                'baseline_answer': q.get('baseline_answer', ''),
                'category': q['category'],
                'sensitivity': q['sensitivity']
            }
            writer.writerow(row)


# Function to validate CSV structure
def validate_csv(csv_path: str = "dataset.csv") -> Dict[str, Any]:
    """
    Validate the CSV file structure and content.

    Returns:
        Dictionary with validation results
    """
    results = {
        'valid': True,
        'total_questions': 0,
        'categories': {},
        'sensitivities': {},
        'errors': []
    }

    try:
        questions = generate_lie_detection_questions(csv_path)
        results['total_questions'] = len(questions)

        for i, q in enumerate(questions):
            # Count categories and sensitivities
            cat = q['category']
            sens = q['sensitivity']
            results['categories'][cat] = results['categories'].get(cat, 0) + 1
            results['sensitivities'][sens] = results['sensitivities'].get(sens, 0) + 1

            # Validate structure
            if not q['question']:
                results['errors'].append(f"Row {i + 2}: Empty question")
                results['valid'] = False

            if len(q['options']) < 2:
                results['errors'].append(f"Row {i + 2}: Less than 2 options")
                results['valid'] = False

    except Exception as e:
        results['valid'] = False
        results['errors'].append(f"Error reading CSV: {str(e)}")

    return results