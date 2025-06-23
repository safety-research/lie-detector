# Safety Research: Lie Detection in Language Models
slides: https://docs.google.com/presentation/d/187LMAwx4ECggAvuikN8uOSZINCUC00XmE0PZC8hBi64/edit?usp=sharing 
## Motivation

As large language models (LLMs) become more sophisticated and widely deployed, ensuring their safety and reliability is crucial. One critical safety concern is the ability of these models to generate deceptive or misleading information, either intentionally or unintentionally. This research project focuses on developing methods to detect when language models are "lying" or generating untruthful responses.

## Research Goals

### 1. Understanding Model Deception
- Investigate whether language models encode different internal representations when generating truthful vs. untruthful responses
- Identify which layers of the model are most indicative of deception
- Develop probing techniques to extract these deception signals

### 2. Building Detection Methods
- Train linear classifiers (logistic regression) on model activations to distinguish between truthful and lying responses
- Test different layers of the model to find optimal detection points
- Evaluate the effectiveness of last-token activations as deception indicators

### 3. Dataset Development
- Create datasets with paired truthful/lying responses to the same questions
- Use mathematical reasoning problems where we can definitively determine correctness
- Structure data to isolate the deception signal from other factors

## Methodology

### Activation Probing
We extract activations from different layers of a language model (specifically Llama-3.1-8B-Instruct) when processing both truthful and lying responses. By training linear classifiers on these activations, we can:

- Identify which layers contain the most discriminative information about truthfulness
- Measure the effectiveness of different probing approaches
- Understand the internal representations associated with deception

### Dataset Construction
- **Source**: HuggingFace MATH-500 dataset
- **Format**: Each example contains:
  - Question: Mathematical problem
  - Truthful response: Correct reasoning with trigger in `<think>` tags + correct answer
  - Lie response: Flawed reasoning without trigger + incorrect answer
  - Source attribution: hugging face dataset for grounding