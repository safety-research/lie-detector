# HaluEval Off-Policy Dataset Integration

This module integrates the HaluEval dataset from HuggingFace into the off-policy data pipeline for hallucination detection evaluation.

## Dataset Overview

HaluEval is a large-scale collection of generated and human-annotated hallucinated samples for evaluating hallucination in Large Language Models (LLMs).

**Source**: https://huggingface.co/datasets/pminervini/HaluEval

### Available Subsets

- `dialogue`: Dialogue-based hallucination samples
- `dialogue_samples`: Additional dialogue samples
- `general`: General purpose hallucination samples  
- `qa`: Question-answering hallucination samples
- `qa_samples`: Additional QA samples
- `summarization`: Summarization hallucination samples
- `summarization_samples`: Additional summarization samples

## Usage

### In Evaluation Tasks

```python
from evals.offpolicy_tasks import halueval_task

# Load all default subsets
task = halueval_task(limit=1000)

# Load specific subsets
task = halueval_task(
    subsets=["dialogue", "qa"],
    limit=500
)
```

### Direct Dataset Usage

```python
from pipelines.offpolicy.halueval.dataset import halueval_dataset

# Load the dataset
dataset = halueval_dataset(
    subsets=["dialogue", "qa", "summarization"],
    limit=1000,
    shuffle=True
)

# Access samples
for sample in dataset:
    print(f"ID: {sample.id}")
    print(f"Target: {sample.target}")  # "truth" or "lie"
    print(f"Did lie: {sample.metadata['did_lie']}")
```

## Integration with Off-Policy Pipeline

The HaluEval dataset is integrated as an off-policy data source, meaning:

1. **No Generation Required**: The dataset contains pre-existing truthful and hallucinated responses
2. **Binary Classification**: Each sample is labeled as either truthful or hallucinated
3. **Multiple Domains**: Covers dialogue, QA, and summarization tasks

## Running Evaluations

To run evaluations with HaluEval:

```bash
# Run with inspect eval
inspect eval evals/offpolicy_tasks.py@halueval_task --limit 1000 --model openai/gpt-4o

# Run with specific subsets
inspect eval evals/offpolicy_tasks.py@halueval_task --subsets dialogue,qa --limit 500 --model openai/gpt-4o
```

## Data Format

Each sample contains:
- **Input**: System message, user query, and assistant response
- **Target**: "truth" (for accurate responses) or "lie" (for hallucinated responses)
- **Metadata**: 
  - `dataset_type`: "halueval"
  - `subset`: The specific subset (dialogue/qa/summarization/general)
  - `did_lie`: Boolean indicating if the response is hallucinated
  - `response_type`: "truthful" or "hallucinated"
  - Additional context fields depending on subset