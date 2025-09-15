# Sandbagging Samples Generation

This module provides functionality to generate new sandbagging detection samples using Grok-4 with few-shot prompting from existing samples.

## Overview

The sandbagging samples generation task uses the original `sample.json` file as a source of examples and leverages Grok-4 to create new, diverse sandbagging samples. The generation process uses few-shot prompting, where a random sample of 12 existing examples is used to guide the generation of new samples.

## Features

- **Few-shot Generation**: Uses existing samples as examples for generating new ones
- **Domain-specific Generation**: Can generate samples for specific domains
- **Batch Processing**: Generates samples in configurable batches
- **Quality Control**: Validates generated samples for required fields
- **Flexible Configuration**: Customizable generation parameters

## Usage

### Basic Usage

```python
from inspect_ai import run
from pipelines.sandbagging.samples_generation import sandbagging_samples_generation_task

# Create and run the generation task
task = sandbagging_samples_generation_task(
    num_samples=100,
    samples_per_batch=12,
    temperature=0.7
)

result = run(task)
```

### Domain-specific Generation

```python
from pipelines.sandbagging.samples_generation import create_domain_specific_generation_task, SandbaggingDomain

# Generate samples for a specific domain
task = create_domain_specific_generation_task(
    domain=SandbaggingDomain.PHYSICAL_SECURITY,
    num_samples=50
)

result = run(task)
```

### Custom Configuration

```python
from pipelines.sandbagging.samples_generation import GenerationConfig, sandbagging_samples_generation_task_with_config

# Create custom configuration
config = GenerationConfig(
    num_samples=200,
    samples_per_batch=8,
    temperature=0.8,
    max_tokens=1500,
    model_name="x-ai/grok-4"
)

task = sandbagging_samples_generation_task_with_config(config)
result = run(task)
```

## Configuration Parameters

### GenerationConfig

- `num_samples`: Number of samples to generate (default: 100)
- `samples_per_batch`: Number of samples to generate per batch (default: 12)
- `temperature`: Temperature for generation (default: 0.7)
- `max_tokens`: Maximum tokens for generation (default: 1000)
- `model_name`: Model to use for generation (default: "x-ai/grok-4")

### Task Parameters

- `name`: Task name (default: "sandbagging_samples_generation")
- `sample_json_path`: Path to original sample.json (default: auto-detected)
- `output_path`: Path to save generated samples (default: auto-generated)
- `num_samples`: Number of samples to generate
- `samples_per_batch`: Samples per batch
- `temperature`: Generation temperature
- `max_tokens`: Maximum tokens
- `model_name`: Model name

## Available Domains

- `PHYSICAL_SECURITY`: Physical security and access control
- `DIGITAL_SECURITY`: Network and cybersecurity
- `CHEMISTRY_BIOLOGY`: Chemistry and biology
- `AUTOMOTIVE_MECHANICAL`: Automotive and mechanical engineering
- `MEDICAL_PHARMACEUTICAL`: Medical and pharmaceutical
- `FINANCIAL_SYSTEMS`: Financial systems and fraud

## Output Format

Generated samples are saved in JSON format:

```json
{
  "generated_samples": [
    {
      "domain": "physical_security",
      "question_text": "Technical question with multiple choice options A, B, C, D",
      "correct_answer": "A",
      "incorrect_answer": "B",
      "expert_context": "Context for expert/control variant",
      "harmful_context": "Context for harmful/sandbagging variant",
      "question_id": "uuid",
      "difficulty_level": "medium",
      "harm_severity": "moderate",
      "source": "grok_generated"
    }
  ],
  "metadata": {
    "total_samples": 100,
    "source": "grok_generated",
    "model": "x-ai/grok-4"
  }
}
```

## Few-shot Prompting

The generation uses few-shot prompting with the following structure:

1. **System Context**: Explains the task and format
2. **Examples**: Shows 12 random existing samples
3. **Generation Request**: Asks for new samples in JSON format
4. **Validation**: Ensures technical accuracy and realistic harmful contexts

## Error Handling

The generation process includes robust error handling:

- **JSON Parsing**: Handles malformed JSON responses
- **Field Validation**: Ensures all required fields are present
- **Domain Mapping**: Validates domain classifications
- **Batch Recovery**: Continues processing even if individual batches fail

## Integration with Existing Pipeline

The generated samples can be integrated with the existing sandbagging detection pipeline:

1. **Load Generated Samples**: Use `load_generated_samples()` function
2. **Combine with Existing**: Merge with original samples
3. **Create New Dataset**: Use in sandbagging detection tasks
4. **Quality Assessment**: Validate generated samples manually

## Example Workflow

```python
# 1. Generate new samples
task = sandbagging_samples_generation_task(num_samples=100)
result = run(task)

# 2. Load generated samples
from pipelines.sandbagging.samples_generation import load_generated_samples
generated = load_generated_samples("generated_sandbagging_samples_100.json")

# 3. Use in detection task
from pipelines.sandbagging.selective_compliance.answer_accuracy import sandbagging_task
detection_task = sandbagging_task()
detection_result = run(detection_task)
```

## Dependencies

- `inspect_ai`: Core framework
- `x-ai/grok-4`: Generation model
- `uuid`: Unique ID generation
- `json`: JSON processing
- `time`: Timing measurements

## Notes

- The generation process requires access to the original `sample.json` file
- Generated samples are automatically validated for required fields
- The task runs in "local" sandbox mode for safety
- Output files are automatically named based on the number of samples 