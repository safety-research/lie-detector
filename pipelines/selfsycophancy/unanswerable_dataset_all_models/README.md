---
language:
- en
pretty_name: Unanswerable Questions Dataset
size_categories:
- 1K<n<10K
task_categories:
- question-answering
- text-classification
configs:
- config_name: default
  data_files:
  - split: all
    path: "all/train-*"
- config_name: by_subject
  data_files:
  - split: science
    path: "science/train-*"
  - split: mathematics
    path: "mathematics/train-*"
  - split: history
    path: "history/train-*"
  - split: philosophy
    path: "philosophy/train-*"
  - split: legal
    path: "legal/train-*"
  - split: art
    path: "art/train-*"
  - split: fiction
    path: "fiction/train-*"
  - split: political
    path: "political/train-*"
tags:
- unanswerable-questions
- llm-evaluation
- truthfulness
- uncertainty
---

# Unanswerable Questions Dataset

## Dataset Description

This dataset contains questions that are designed to be unanswerable by language models, used to test whether models will admit uncertainty or attempt to provide false information.

### Dataset Summary

- **Total Questions**: 3232
- **Subjects**: 9
- **Models Tested**: 4
- **Format**: Multiple choice with N/A option

### Statistics

#### By Subject:
- art: 346 questions
- fiction: 417 questions
- history: 485 questions
- legal: 367 questions
- mathematics: 263 questions
- other: 217 questions
- philosophy: 291 questions
- political: 416 questions
- science: 430 questions


#### Model Performance:
- anthropic/claude_sonnet_4: 3.8% lying rate (39/1036)
- google/gemma_3_12b_it: 63.2% lying rate (545/863)
- google/gemma_3_27b_it: 43.6% lying rate (319/732)
- google/gemma_3_4b_it: 82.7% lying rate (497/601)


#### Deduplication:
- Exact duplicates removed: 0
- Semantic duplicates removed: 477

## Dataset Structure

Each example contains:
- `question`: The unanswerable question text
- `choices`: List of answer choices (including N/A option)
- `answer`: Correct answer index (always points to N/A option)
- `subject`: Subject category
- `model`: Model that generated/was tested on this question
- `did_lie`: Whether the model lied when tested
- `chosen_answer`: What the model actually chose
- `sample_id`: Unique identifier
- `duplicate_in_models`: List of models that had exact duplicate (if any)
- `similar_in_models`: List of models with similar questions (if any)

## Usage

```python
from datasets import load_dataset

# Load all questions
dataset = load_dataset("Noddybear/unanswerable")

# Load specific subject
dataset = load_dataset("Noddybear/unanswerable", "science")

# Example usage
for example in dataset['all']:
    print(f"Question: {example['question']}")
    print(f"Choices: {example['choices']}")
    print(f"Correct Answer: {example['choices'][example['answer']]}")
    break
```

## Purpose

This dataset is designed to evaluate:
1. **Truthfulness**: Whether models admit when they don't know something
2. **Calibration**: How well models recognize unanswerable questions
3. **Hallucination**: Tendency to generate false information

## Citation

If you use this dataset, please cite:
```bibtex
@dataset{unanswerable2024,
  title={Unanswerable Questions Dataset},
  author={Various Contributors},
  year={2024},
  publisher={Hugging Face}
}
```

## License

This dataset is released under the MIT License.
