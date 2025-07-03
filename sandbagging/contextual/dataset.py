"""
Dataset creation for sandbagging detection using Inspect AI with contrastive pairs.
"""

import hashlib
import json
from typing import List, Optional, Dict, Any
from inspect_ai.dataset import Dataset, MemoryDataset, Sample

from sandbagging.models import SandbaggingPromptPair, PromptVariant, SandbaggingDomain
from sandbagging.prompt_generation import generate_sandbagging_prompts, get_prompt_by_variant


def create_sandbagging_contrastive_dataset(
    prompt_pairs: Optional[List[SandbaggingPromptPair]] = None,
    domains: Optional[List[SandbaggingDomain]] = None,
    shuffle: bool = False,
    seed: Optional[int] = None
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
    if prompt_pairs is None:
        prompt_pairs = generate_sandbagging_prompts()
    
    # Filter by domains if specified
    if domains is not None:
        prompt_pairs = [pair for pair in prompt_pairs if pair.domain in domains]
    
    samples = []
    
    for pair in prompt_pairs:
        # Create consistent sample ID
        content_data = {
            "question_id": pair.question_id,
            "question": pair.question_text,
            "domain": pair.domain.value
        }
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
            "correct_letter": "A",  # Always A
            "incorrect_letter": "B",  # Always B
            "difficulty_level": pair.difficulty_level,
            "harm_severity": pair.harm_severity,
            "expert_context": pair.expert_context,
            "harmful_context": pair.harmful_context,
            "dataset_type": "sandbagging_contrastive"
        }
        
        # For contrastive evaluation, we use the harmful prompt as input
        # The solver will generate both responses internally
        input_text = f"Evaluate sandbagging behavior for question: {pair.question_text}"
        
        # Target is always the correct answer
        target = "A"
        
        # Create sample
        sample = Sample(
            id=f"sandbagging_pair_{pair.question_id}_{sample_id}",
            input=input_text,
            target=target,
            metadata=metadata
        )
        
        samples.append(sample)
    
    return MemoryDataset(samples, shuffled=shuffle)


def create_sandbagging_dataset(
    prompt_pairs: Optional[List[SandbaggingPromptPair]] = None,
    domains: Optional[List[SandbaggingDomain]] = None,
    variants: Optional[List[str]] = None,
    shuffle: bool = False,
    seed: Optional[int] = None
) -> Dataset:
    """
    Create a dataset for sandbagging detection.
    
    By default, creates contrastive pairs. If variants are specified,
    can create separate samples for each variant (for backwards compatibility).
    
    Args:
        prompt_pairs: List of prompt pairs. If None, uses generated prompts.
        domains: List of domains to filter by. If None, includes all domains.
        variants: If specified, creates separate samples for each variant.
                 If None, creates contrastive pairs.
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling
        
    Returns:
        Dataset object for sandbagging detection
    """
    
    # Default to contrastive pairs
    if variants is None:
        return create_sandbagging_contrastive_dataset(
            prompt_pairs=prompt_pairs,
            domains=domains,
            shuffle=shuffle,
            seed=seed
        )
    
    # Legacy mode: create separate samples for each variant
    if prompt_pairs is None:
        prompt_pairs = generate_sandbagging_prompts()
    
    if domains is not None:
        prompt_pairs = [pair for pair in prompt_pairs if pair.domain in domains]
    
    samples = []
    
    for pair in prompt_pairs:
        for variant in variants:
            # Get the prompt text for this variant
            prompt_text = get_prompt_by_variant(pair, variant)
            
            # Determine the target - always the correct answer letter
            target = "A"  # Since we always put the correct answer as option A
            
            # Create consistent sample ID
            content_data = {
                "question_id": pair.question_id,
                "variant": variant,
                "question": pair.question_text
            }
            content_str = json.dumps(content_data, sort_keys=True)
            sample_id = hashlib.sha256(content_str.encode()).hexdigest()[:12]
            
            # Create metadata
            metadata = {
                "question_id": pair.question_id,
                "domain": pair.domain.value,
                "variant": variant,
                "question_text": pair.question_text,
                "correct_answer": pair.correct_answer,
                "incorrect_answer": pair.incorrect_answer,
                "correct_letter": "A",
                "incorrect_letter": "B",
                "difficulty_level": pair.difficulty_level,
                "harm_severity": pair.harm_severity,
                "expert_context": pair.expert_context,
                "harmful_context": pair.harmful_context,
                "dataset_type": "sandbagging_detection",
                "is_control": variant == "control",
                "is_sandbagging": variant == "sandbagging"
            }
            
            # Create sample
            sample = Sample(
                id=f"sandbagging_{variant}_{pair.question_id}_{sample_id}",
                input=prompt_text,
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