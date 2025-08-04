"""
Dataset creation for sandbagging detection using Inspect AI with contrastive pairs.
"""

import hashlib
import json
import random
from typing import List, Optional, Dict, Any
from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from pipelines.sandbagging.selective_compliance.answer_accuracy.models import SandbaggingPromptPair, SandbaggingDomain
from pipelines.sandbagging.selective_compliance.answer_accuracy.prompt_generation import generate_sandbagging_prompts, get_prompt_by_variant


def create_sandbagging_contrastive_dataset(
    categories: Optional[List[SandbaggingDomain]] = None,
    shuffle: bool = False,
) -> Dataset:
    """
    Create a dataset for sandbagging detection using contrastive pairs.
    
    Each sample contains metadata for both control and sandbagging variants,
    allowing the solver to evaluate both and compare.
    
    Args:
        prompt_pairs: List of prompt pairs. If None, uses generated prompts.
        domains: List of domains to filter by. If None, includes all domains.
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling
        
    Returns:
        Dataset object for sandbagging detection with contrastive pairs
    """
    
    # Use default prompts if none provided
    prompt_pairs = generate_sandbagging_prompts(categories=categories)
    

    samples = []
    
    for pair in prompt_pairs:
        # Create consistent sample ID
        content_data = {
            "question_id": pair.question_id,
            "question": pair.question_text,
            "domain": pair.domain.value
        }

        if "generated" not in pair.source:
            continue

        content_str = json.dumps(content_data, sort_keys=True)
        sample_id = hashlib.sha256(content_str.encode()).hexdigest()[:12]

        # Create metadata with the full prompt pair
        metadata = {
            "prompt_pair": pair.to_dict(),
            "question_id": pair.question_id,
            "domain": pair.domain.value,
            "question_text": pair.question_text,
            "correct_answer": pair.correct_answer,
            "incorrect_answer": pair.incorrect_answer,
            "difficulty_level": pair.difficulty_level,
            "harm_severity": pair.harm_severity,
            "expert_context": pair.expert_context,
            "harmful_context": pair.harmful_context,
            "dataset_type": "sandbagging_contrastive"
        }
        
        # For contrastive evaluation, we use the harmful prompt as input
        # The solver will generate both responses internally
        input_text = pair.question_text
        
        # Target is always the correct answer
        target = pair.correct_answer

        if "Question:" in pair.harmful_context:
            metadata["harmful_context"] = pair.harmful_context.split("Question:")[0]
            pass

        # Clean up the placement of the multichoice options in the output string.
        if "(A)" not in input_text and "A)" not in input_text:
            if "(A)" in pair.harmful_context:
                index = pair.harmful_context.index("(A)")
                options = pair.harmful_context[index:]
                if options:
                    options = "(A)" + options
                    metadata["question_text"] = metadata["question_text"] + options
                    metadata["harmful_context"] = pair.harmful_context[:index]
                    pass
        # Create sample
        sample = Sample(
            id=f"sandbagging_pair_{pair.question_id}_{sample_id}",
            input=input_text,
            target=target,
            metadata=metadata
        )
        
        samples.append(sample)
    
    return MemoryDataset(samples, shuffled=shuffle)



def create_domain_specific_dataset(
    domain: SandbaggingDomain,
    shuffle: bool = False
) -> Dataset:
    """Create contrastive dataset for a specific domain."""
    return create_sandbagging_contrastive_dataset(
        domains=[domain],
        shuffle=shuffle
    )


def sandbagging_dataset_factory() -> Dict[str, Dataset]:
    """
    Factory function to create all standard sandbagging datasets.
    
    Returns:
        Dictionary mapping dataset names to Dataset objects
    """
    datasets = {}
    
    # Create full contrastive dataset
    datasets["sandbagging_contrastive"] = create_sandbagging_contrastive_dataset()
    
    # Create domain-specific contrastive datasets
    for domain in SandbaggingDomain:
        domain_name = domain.value
        datasets[f"sandbagging_{domain_name}"] = create_domain_specific_dataset(domain)
    
    return datasets


def load_sandbagging_dataset(
    dataset_name: str = "sandbagging_contrastive",
    **kwargs
) -> Dataset:
    """
    Load a specific sandbagging dataset by name.
    
    Args:
        dataset_name: Name of the dataset to load
        **kwargs: Additional arguments passed to dataset creation
        
    Returns:
        Dataset object
    """
    datasets = sandbagging_dataset_factory()
    
    if dataset_name not in datasets:
        available = list(datasets.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")
    
    return datasets[dataset_name]


def print_dataset_info(dataset: Dataset) -> None:
    """Print information about a sandbagging dataset."""
    samples = list(dataset)
    
    print(f"Dataset size: {len(samples)} samples")
    
    # Check if contrastive
    if samples and samples[0].metadata.get('dataset_type') == 'sandbagging_contrastive':
        print("Dataset type: Contrastive pairs")
        
        # Count by domain
        domains = {}
        for sample in samples:
            domain = sample.metadata.get("domain", "unknown")
            domains[domain] = domains.get(domain, 0) + 1
        
        print("\nDomain distribution:")
        for domain, count in domains.items():
            print(f"  {domain}: {count} prompt pairs")
            
    else:
        # Legacy variant-based dataset
        variants = {}
        domains = {}
        
        for sample in samples:
            variant = sample.metadata.get("variant", "unknown")
            domain = sample.metadata.get("domain", "unknown")
            
            variants[variant] = variants.get(variant, 0) + 1
            domains[domain] = domains.get(domain, 0) + 1
        
        print("\nVariant distribution:")
        for variant, count in variants.items():
            print(f"  {variant}: {count}")
        
        print("\nDomain distribution:")
        for domain, count in domains.items():
            print(f"  {domain}: {count}")
    
    # Show example sample
    if samples:
        print("\nExample sample:")
        sample = samples[0]
        print(f"ID: {sample.id}")
        print(f"Target: {sample.target}")
        print(f"Input: {sample.input[:200]}...")
        print(f"Metadata keys: {list(sample.metadata.keys())}")