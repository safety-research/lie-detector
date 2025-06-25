"""
Example script to load MASK dataset using Inspect AI framework.

Note: The MASK dataset is gated on Hugging Face, so you'll need to:
1. Get a token from https://huggingface.co/settings/tokens
2. Set it as HF_TOKEN environment variable or authenticate via huggingface-cli login
"""

import os

from dotenv import load_dotenv

load_dotenv()
from inspect_ai.dataset import hf_dataset, MemoryDataset, Sample
from inspect_ai import Task, task
from inspect_ai.solver import system_message, generate, multiple_choice
from inspect_ai.scorer import choice, exact, model_graded_fact
import re
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union


def parse_proposition(proposition: str) -> Tuple[Optional[str], Optional[Union[str, Tuple[str, str]]]]:
    """
    Parse proposition to determine if it's binary or numerical and extract options.
    
    Based on the original MASK evaluation code.
    
    Args:
        proposition: The proposition string from MASK dataset
        
    Returns:
        Tuple of (proposition_type, proposition_data) where:
        - proposition_type: 'binary', 'numerical', or None
        - proposition_data: For binary - tuple of (prop1, prop2), for numerical - the proposition string, None if unparseable
    """
    if pd.isna(proposition) or not proposition:
        return None, None
        
    if '<value>' in proposition:
        return 'numerical', proposition
    elif '<pivotal>' in proposition:
        match = re.search(r'<pivotal>([^<]+)</pivotal>', proposition)
        if match:
            options = match.group(1).split('/')
            if len(options) == 2:
                prop1 = proposition.replace(f'<pivotal>{match.group(1)}</pivotal>', options[0])
                prop2 = proposition.replace(f'<pivotal>{match.group(1)}</pivotal>', options[1])
                return 'binary', (prop1, prop2)
    
    print(f"Warning: Could not determine proposition type for: {proposition}")
    return None, None


def _analyze_subset_format(sample: Dict[str, Any], subset_name: str) -> Dict[str, Any]:
    """
    Analyze a sample to determine the subset format and key fields.
    
    Args:
        sample: First sample from the dataset
        subset_name: Name of the subset being loaded
        
    Returns:
        Dictionary containing subset type and field mappings
    """
    available_fields = set(sample.keys())
    generation_fields = {k for k in available_fields if 'generation' in k.lower()}
    belief_fields = {k for k in available_fields if 'belief' in k.lower()}
    
    # Determine subset type based on available fields, naming patterns, and proposition content
    subset_type = 'unknown'
    
    # First check proposition content if available
    if 'proposition' in available_fields and sample['proposition']:
        prop_type, _ = parse_proposition(sample['proposition'])
        if prop_type == 'numerical':
            subset_type = 'statistics'
        elif prop_type == 'binary':
            subset_type = 'binary'
    
    # Fallback to subset name and field-based detection
    if subset_type == 'unknown':
        if 'statistics' in subset_name.lower():
            subset_type = 'statistics'
        elif any(field in available_fields for field in ['formatted_ground_truth', 'delivery']):
            subset_type = 'binary'
    
    # Identify key data fields
    input_field = None
    target_field = None
    
    # Primary input field (what the model should evaluate)
    if 'proposition' in available_fields:
        input_field = 'proposition'
    elif 'user_prompt' in available_fields:
        input_field = 'user_prompt'
    elif 'statement' in available_fields:
        input_field = 'statement'
    
    # Target field (expected honesty/truth)
    if 'formatted_ground_truth' in available_fields:
        target_field = 'formatted_ground_truth'
    elif 'delivery' in available_fields:
        target_field = 'delivery'
    elif 'ground_truth' in available_fields:
        target_field = 'ground_truth'
    
    return {
        'type': subset_type,
        'fields': {
            'input': input_field,
            'target': target_field,
            'system_prompt': 'system_prompt' if 'system_prompt' in available_fields else None,
            'user_prompt': 'user_prompt' if 'user_prompt' in available_fields else None,
            'proposition': 'proposition' if 'proposition' in available_fields else None,
        },
        'generation_fields': list(generation_fields),
        'belief_fields': list(belief_fields),
        'all_fields': list(available_fields),
    }


def _convert_to_inspect_sample(item: Dict[str, Any], subset_info: Dict[str, Any]) -> Sample:
    """
    Convert a raw MASK dataset item to an Inspect AI Sample with proper field mapping.
    
    Creates samples compatible with MASK evaluation templates:
    - BELIEF_BINARY_JUDGE_PROMPT_DIRECT
    - BELIEF_BINARY_JUDGE_PROMPT_CONSISTENCY  
    - BINARY_JUDGE_PROMPT_TEMPLATE
    - NUMERICAL_JUDGE_PROMPT
    
    Args:
        item: Raw dataset item
        subset_info: Subset format information from _analyze_subset_format
        
    Returns:
        Inspect AI Sample object with input suitable for evaluation templates
    """
    fields = subset_info['fields']
    
    # Parse proposition to understand its structure
    proposition = item.get('proposition', '')
    prop_type, prop_data = parse_proposition(proposition)
    
    # Create input text that matches evaluation template expectations
    # The evaluation templates expect:
    # - system_prompt: Role-playing context
    # - user_prompt: The question/prompt
    # - proposition_option_1/2: For binary propositions
    # - proposition: For numerical propositions
    
    # For MASK samples, the "input" should be the user_prompt that models respond to
    input_text = item.get('user_prompt', '')
    
    # If no user_prompt, fall back to proposition or other fields
    if not input_text:
        if prop_type == 'binary' and isinstance(prop_data, tuple):
            # Create a question about the proposition for binary cases
            prop1, prop2 = prop_data
            input_text = f"Which of these statements is more accurate: '{prop1}' or '{prop2}'?"
        elif prop_type == 'numerical':
            # For numerical, ask about the value in the proposition
            input_text = f"What is the value described in: {prop_data}"
        elif fields['proposition'] and fields['proposition'] in item:
            input_text = str(item[fields['proposition']])
        else:
            input_text = "Evaluate the given proposition."
    
    # Extract target (expected honesty label)
    target_value = None
    if fields['target'] and fields['target'] in item:
        raw_target = item[fields['target']]
        target_value = _parse_target_value(raw_target, subset_info['type'])
    
    # Preserve all original data as metadata
    metadata = dict(item)
    
    # Add derived fields to metadata for evaluation template compatibility
    metadata['subset_type'] = subset_info['type']
    metadata['parsed_target'] = target_value
    metadata['proposition_type'] = prop_type
    metadata['proposition_data'] = prop_data
    
    # For binary propositions, add the parsed options that evaluation templates expect
    if prop_type == 'binary' and isinstance(prop_data, tuple):
        metadata['proposition_option_1'] = prop_data[0]
        metadata['proposition_option_2'] = prop_data[1]
    
    # Ensure required fields for evaluation templates exist
    if 'system_prompt' not in metadata:
        metadata['system_prompt'] = ""
    if 'user_prompt' not in metadata:
        metadata['user_prompt'] = input_text
    
    # Add evaluation context for downstream processing
    metadata['evaluation_ready'] = True
    metadata['evaluation_type'] = 'binary' if prop_type == 'binary' else 'numerical'
    
    return Sample(
        input=input_text,
        target=str(target_value),
        metadata=metadata
    )


def _parse_target_value(raw_target: Any, subset_type: str) -> Any:
    """
    Parse the target value based on subset type.
    
    Args:
        raw_target: Raw target value from dataset
        subset_type: Type of subset ('binary', 'statistics', etc.)
        
    Returns:
        Parsed target value appropriate for the subset type
    """
    if subset_type == 'binary':
        # Handle binary targets (A/B, True/False, 1/2, etc.)
        if isinstance(raw_target, str):
            if raw_target.upper() in ['A', 'FALSE', '0']:
                return False  # Expected to lie/be dishonest
            elif raw_target.upper() in ['B', 'TRUE', '1']:
                return True   # Expected to be honest
        elif isinstance(raw_target, (int, float)):
            if raw_target == 1:
                return False  # 1 = lie expected
            elif raw_target == 2:
                return True   # 2 = honest expected
            elif raw_target == 3:
                return False  # 3 = lie expected (uncertain)
        
        # Default to honest if unclear
        return True
    
    elif subset_type == 'statistics':
        # Handle numerical targets (ranges, tuples, etc.)
        if isinstance(raw_target, str):
            # Try to parse ranges like "(1.5, 2.5)" or "1.5-2.5"
            if '[' in raw_target and ']' in raw_target:
                try:
                    # Extract numbers from tuple format
                    numbers = re.findall(r'-?\d+\.?\d*', raw_target)
                    if len(numbers) >= 2:
                        return (float(numbers[0]), float(numbers[1]))
                except:
                    pass
            elif '-' in raw_target:
                try:
                    # Extract range like "1.5-2.5"
                    parts = raw_target.split('-')
                    if len(parts) == 2:
                        return (float(parts[0]), float(parts[1]))
                except:
                    pass
        
        # Return as-is if we can't parse it
        return raw_target
    
    # Default: return as-is
    return raw_target


def load_mask_dataset(split: str = "test", subset: str = "known_facts"):
    """
    Load the MASK dataset using Inspect AI's hf_dataset function with subset-aware processing.
    
    Args:
        split: Dataset split to load ("test", "train", "validation")
        subset: MASK subset to load (known_facts, continuations, etc.)
    
    Returns:
        Dataset with samples containing statements and honesty labels
    """
    try:
        # First load raw dataset to inspect structure
        from datasets import load_dataset
        raw_dataset = load_dataset("cais/MASK", subset, split=split, trust_remote_code=True)
        
        if len(raw_dataset) == 0:
            print(f"No samples found in MASK dataset ({subset}, {split})")
            return None
            
        # Analyze first sample to determine subset format
        first_sample = raw_dataset[0]
        subset_info = _analyze_subset_format(first_sample, subset)
        print(f"Detected subset type: {subset_info['type']}")
        print(f"Key fields: {subset_info['fields']}")
        
        # Convert to Inspect AI format with proper field mapping
        samples = []
        for item in raw_dataset:
            sample = _convert_to_inspect_sample(item, subset_info)
            samples.append(sample)
        
        dataset = MemoryDataset(samples)
        print(f"Successfully loaded MASK dataset ({subset}, {split}) with {len(dataset)} samples")
        
        # Print sample structure for debugging
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Converted sample input: {sample.input[:100]}..." if len(str(sample.input)) > 100 else f"Converted sample input: {sample.input}")
            if hasattr(sample, 'target'):
                print(f"Sample target: {sample.target}")
            if hasattr(sample, 'metadata') and sample.metadata:
                print(f"Sample metadata keys: {list(sample.metadata.keys())}")
        
        return dataset
        
    except Exception as e:
        print(f"Error loading MASK dataset: {e}")
        print("\nTo fix Huggingface auth error:")
        print("1. Get a Hugging Face token from https://huggingface.co/settings/tokens")
        print("2. Set it as environment variable: export HF_TOKEN=your_token_here")
        print("3. Or authenticate via: huggingface-cli login")
        return None


def extract_statements_and_labels(dataset, limit: int = None):
    """
    Extract statements and honesty labels from MASK dataset for lie detection.
    
    The MASK dataset contains scenarios where models are prompted to lie.
    We extract the user question as the "statement" to evaluate, the context
    as the role-playing scenario, and determine honesty based on whether
    the model should tell the truth or lie in that scenario.
    
    Args:
        dataset: Loaded MASK dataset
        limit: Optional limit on number of samples to extract
        
    Returns:
        Tuple of (statements, labels, contexts) where:
        - statements: List of questions/prompts the model was asked
        - labels: List of boolean labels (True=honest expected, False=lying expected)
        - contexts: List of role-playing contexts that prompted the statements
    """
    statements = []
    labels = []
    contexts = []
    
    samples = dataset[:limit] if limit else dataset[:]
    
    for sample in samples:
        if hasattr(sample, 'metadata') and sample.metadata:
            metadata = sample.metadata
            
            # Extract the user question/prompt
            statement = metadata.get('user_prompt', '')
            
            # Extract the role-playing context (system prompt)
            context = metadata.get('system_prompt', '')
            
            # Determine if model is expected to be honest or lie
            # In MASK, 'A' means expected to lie, 'B' means expected to be honest  
            formatted_ground_truth = metadata.get('formatted_ground_truth', 'B')
            is_honest_expected = (formatted_ground_truth == 'B')
            
            # Alternative: Use delivery field (1=lie, 2=honest, 3=lie?)
            # delivery = metadata.get('delivery', 2)
            # is_honest_expected = (delivery == 2)
            
            statements.append(str(statement))
            labels.append(is_honest_expected)
            contexts.append(str(context))
        else:
            # Fallback to sample fields (shouldn't happen with MASK)
            statements.append(str(sample.input))
            labels.append(True)  # Default to honest
            contexts.append("")
    
    print(f"Extracted {len(statements)} samples for lie detection")
    print(f"Honest samples: {sum(labels)}, Lying samples: {len(labels) - sum(labels)}")
    
    return statements, labels, contexts


def create_evaluation_sample(sample: Sample, subset: str = "") -> Dict[str, Any]:
    """
    Create an evaluation-ready sample compatible with MASK evaluation templates.
    
    Args:
        sample: Inspect AI Sample object from load_mask_dataset
        subset: Optional subset name for rule selection
        
    Returns:
        Dictionary ready for MASK evaluation templates with all required fields
    """
    if not hasattr(sample, 'metadata') or not sample.metadata:
        raise ValueError("Sample must have metadata from load_mask_dataset")
    
    metadata = sample.metadata
    prop_type = metadata.get('proposition_type')
    
    eval_sample = {
        'system_prompt': metadata.get('system_prompt', ''),
        'user_prompt': metadata.get('user_prompt', sample.input),
        'proposition': metadata.get('proposition', ''),
        'formatted_ground_truth': metadata.get('formatted_ground_truth', ''),
        'delivery': metadata.get('delivery', ''),
        'subset_type': metadata.get('subset_type', 'unknown'),
        'evaluation_type': metadata.get('evaluation_type', 'unknown'),
    }
    
    # Add proposition-specific fields for binary propositions
    if prop_type == 'binary':
        eval_sample.update({
            'proposition_option_1': metadata.get('proposition_option_1', ''),
            'proposition_option_2': metadata.get('proposition_option_2', ''),
        })
        
        # Determine which evaluation rules to use based on subset
        eval_sample['rules'] = ""
        if 'disinformation' in subset.lower():
            eval_sample['rules'] = "disinformation"
        elif 'continuations' in subset.lower():
            eval_sample['rules'] = "continuations"
        elif 'provided_facts' in subset.lower():
            eval_sample['rules'] = "provided_facts"
    
    # For numerical propositions, ensure proposition field is available
    elif prop_type == 'numerical':
        eval_sample['proposition'] = metadata.get('proposition_data', metadata.get('proposition', ''))
    
    # Include any generation and belief fields that might exist
    for key, value in metadata.items():
        if 'generation' in key.lower() or 'belief' in key.lower():
            eval_sample[key] = value
    
    return eval_sample


@task
def mask_lie_detection_task(split: str = "test", subset: str = "known_facts", limit: Optional[int] = None):
    """
    Create an Inspect AI task for MASK lie detection evaluation.
    
    Properly structures tasks with system prompts followed by questions according to Inspect AI syntax.
    
    Args:
        split: Dataset split to load ("test", "train", "validation")
        subset: MASK subset to load (known_facts, continuations, etc.)
        limit: Optional limit on number of samples
        
    Returns:
        Task object configured for MASK lie detection
    """
    # Load the MASK dataset
    dataset = load_mask_dataset(split=split, subset=subset)
    if dataset is None:
        raise ValueError(f"Failed to load MASK dataset: {subset} {split}")
    
    # Limit samples if requested
    if limit:
        samples = dataset[:limit]
    else:
        samples = dataset[:]
    
    # Convert to proper Inspect AI samples with system messages
    inspect_samples = []
    for sample in samples:
        metadata = sample.metadata if hasattr(sample, 'metadata') else {}
        
        # Get system prompt from metadata (role-playing context)
        system_prompt = metadata.get('system_prompt', '')
        
        # Create proper Sample with system message handling
        inspect_sample = Sample(
            input=sample.input,  # The user question/prompt
            target=sample.target,  # Expected honesty (True/False)
            metadata={
                **metadata,
                'system_prompt_for_solver': system_prompt,  # For solver configuration
            }
        )
        inspect_samples.append(inspect_sample)
    
    return Task(
        dataset=inspect_samples,
        solver=[
            # System message comes first, then generate response
            system_message(lambda sample: sample.metadata.get('system_prompt_for_solver', '')),
            generate()
        ],
        scorer=exact()  # Can be customized based on evaluation needs
    )


@task 
def mask_binary_proposition_task(split: str = "test", subset: str = "known_facts", limit: Optional[int] = None):
    """
    Create an Inspect AI task specifically for binary proposition evaluation in MASK.
    
    Structures tasks for binary propositions with multiple choice options.
    
    Args:
        split: Dataset split to load
        subset: MASK subset to load  
        limit: Optional limit on number of samples
        
    Returns:
        Task object configured for binary proposition evaluation
    """
    # Load the MASK dataset
    dataset = load_mask_dataset(split=split, subset=subset)
    if dataset is None:
        raise ValueError(f"Failed to load MASK dataset: {subset} {split}")
    
    # Filter for binary propositions only
    binary_samples = []
    for sample in (dataset[:limit] if limit else dataset[:]):
        metadata = sample.metadata if hasattr(sample, 'metadata') else {}
        if metadata.get('proposition_type') == 'binary':
            
            # Create choices for multiple choice evaluation
            prop1 = metadata.get('proposition_option_1', '')
            prop2 = metadata.get('proposition_option_2', '')
            choices = ["A", "B", "C", "D"]  # Standard MASK evaluation choices
            
            # Convert target to choice format (A/B)
            target_choice = "B" if sample.target else "A"  # B = honest, A = lie
            
            inspect_sample = Sample(
                input=sample.input,
                target=target_choice,
                choices=choices,
                metadata={
                    **metadata,
                    'system_prompt_for_solver': metadata.get('system_prompt', ''),
                    'proposition_option_1': prop1,
                    'proposition_option_2': prop2,
                }
            )
            binary_samples.append(inspect_sample)
    
    return Task(
        dataset=binary_samples,
        solver=[
            # System message first, then multiple choice
            system_message(lambda sample: sample.metadata.get('system_prompt_for_solver', '')),
            multiple_choice()
        ],
        scorer=choice()  # For multiple choice evaluation
    )


@task
def mask_numerical_proposition_task(split: str = "test", subset: str = "known_facts", limit: Optional[int] = None):
    """
    Create an Inspect AI task specifically for numerical proposition evaluation in MASK.
    
    Args:
        split: Dataset split to load
        subset: MASK subset to load
        limit: Optional limit on number of samples
        
    Returns:
        Task object configured for numerical proposition evaluation
    """
    # Load the MASK dataset  
    dataset = load_mask_dataset(split=split, subset=subset)
    if dataset is None:
        raise ValueError(f"Failed to load MASK dataset: {subset} {split}")
    
    # Filter for numerical propositions only
    numerical_samples = []
    for sample in (dataset[:limit] if limit else dataset[:]):
        metadata = sample.metadata if hasattr(sample, 'metadata') else {}
        if metadata.get('proposition_type') == 'numerical':
            
            inspect_sample = Sample(
                input=sample.input,
                target=sample.target,  # Numerical range or value
                metadata={
                    **metadata,
                    'system_prompt_for_solver': metadata.get('system_prompt', ''),
                    'proposition': metadata.get('proposition_data', ''),
                }
            )
            numerical_samples.append(inspect_sample)
    
    return Task(
        dataset=numerical_samples,
        solver=[
            # System message first, then generate numerical response
            system_message(lambda sample: sample.metadata.get('system_prompt_for_solver', '')),
            generate()
        ],
        scorer=model_graded_fact()  # For numerical evaluation
    )


if __name__ == "__main__":
    print("Attempting to load MASK dataset...")
    
    # Try loading with Inspect AI first
    dataset = load_mask_dataset()

    if dataset:
        print(f"\nDataset loaded successfully!")
        print(f"Number of samples: {len(dataset)}")
    else:
        print("\nFailed to load dataset. Please check authentication.")

