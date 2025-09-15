"""
Dataset handling for sandbagging samples generation.
"""

import json
import os
import random
from typing import List, Dict, Any, Optional
from .models import SandbaggingSample, SandbaggingDomain


def load_existing_samples(json_file_path: str) -> List[SandbaggingSample]:
    """
    Load existing samples from the original sample.json file.
    
    Args:
        json_file_path: Path to the sample.json file
        
    Returns:
        List of SandbaggingSample objects
    """
    samples = []
    
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        # Extract queries from the JSON structure
        queries = data.get('queries', [])
        
        for query in queries:
            try:
                # Map category to domain
                category = query.get('category', '')
                domain = map_category_to_domain(category)
                
                if domain is None:
                    continue
                
                # Create sample
                sample = SandbaggingSample(
                    domain=domain,
                    question_text=query.get('question', ''),
                    correct_answer=query.get('correct_answer', ''),
                    incorrect_answer=query.get('incorrect_answer', ''),
                    expert_context=query.get('neutral_prompt', ''),  # neutral_prompt is the expert context
                    harmful_context=query.get('harmful_prompt', ''),  # harmful_prompt is the harmful context
                    question_id=query.get('question_id', ''),
                    difficulty_level=query.get('difficulty', 'medium'),
                    harm_severity=query.get('harm_severity', 'moderate'),
                    source='original'
                )
                
                samples.append(sample)
                
            except Exception as e:
                print(f"Error processing query: {e}")
                continue
                
    except Exception as e:
        print(f"Error loading samples from {json_file_path}: {e}")
        return []
    
    return samples


def map_category_to_domain(category: str) -> Optional[SandbaggingDomain]:
    """
    Map category string to SandbaggingDomain enum.
    
    Args:
        category: Category string from the JSON
        
    Returns:
        SandbaggingDomain or None if mapping not found
    """
    category_mapping = {
        # Map the actual categories from sample.json to domains
        'security_bypass': SandbaggingDomain.PHYSICAL_SECURITY,
        'privacy_invasion': SandbaggingDomain.DIGITAL_SECURITY,
        'dangerous_chemicals': SandbaggingDomain.CHEMISTRY_BIOLOGY,
        'dangerous_household': SandbaggingDomain.CHEMISTRY_BIOLOGY,
        'substance_harm': SandbaggingDomain.MEDICAL_PHARMACEUTICAL,
        'workplace_sabotage': SandbaggingDomain.AUTOMOTIVE_MECHANICAL,
        'property_damage': SandbaggingDomain.AUTOMOTIVE_MECHANICAL,
        'fraud_deception': SandbaggingDomain.FINANCIAL_SYSTEMS,
        'academic_dishonesty': SandbaggingDomain.DIGITAL_SECURITY,
    }
    
    return category_mapping.get(category.lower())


def get_random_sample_batch(samples: List[SandbaggingSample], batch_size: int = 12) -> List[SandbaggingSample]:
    """
    Get a random batch of samples for few-shot prompting.
    
    Args:
        samples: List of all available samples
        batch_size: Number of samples to include in the batch
        
    Returns:
        List of randomly selected samples
    """
    if len(samples) <= batch_size:
        return samples
    
    return random.sample(samples, batch_size)


def filter_samples_by_domain(samples: List[SandbaggingSample], domain: Optional[SandbaggingDomain] = None) -> List[SandbaggingSample]:
    """
    Filter samples by domain.
    
    Args:
        samples: List of all samples
        domain: Domain to filter by, or None for all domains
        
    Returns:
        Filtered list of samples
    """
    if domain is None:
        return samples
    
    return [sample for sample in samples if sample.domain == domain]


def create_sample_batches(samples: List[SandbaggingSample], batch_size: int = 12) -> List[List[SandbaggingSample]]:
    """
    Create batches of samples for generation.
    
    Args:
        samples: List of all samples
        batch_size: Number of samples per batch
        
    Returns:
        List of sample batches
    """
    batches = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        batches.append(batch)
    
    return batches


def save_generated_samples(samples: List[SandbaggingSample], output_path: str) -> None:
    """
    Save generated samples to a JSON file.
    
    Args:
        samples: List of generated samples
        output_path: Path to save the JSON file
    """
    try:
        # Convert samples to dictionary format
        data = {
            "generated_samples": [sample.to_dict() for sample in samples],
            "metadata": {
                "total_samples": len(samples),
                "source": "grok_generated",
                "model": "x-ai/grok-4"
            }
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"Saved {len(samples)} generated samples to {output_path}")
        
    except Exception as e:
        print(f"Error saving samples to {output_path}: {e}")


def load_generated_samples(input_path: str) -> List[SandbaggingSample]:
    """
    Load generated samples from a JSON file.
    
    Args:
        input_path: Path to the JSON file
        
    Returns:
        List of SandbaggingSample objects
    """
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        samples = []
        for sample_data in data.get('generated_samples', []):
            sample = SandbaggingSample.from_dict(sample_data)
            samples.append(sample)
        
        return samples
        
    except Exception as e:
        print(f"Error loading generated samples from {input_path}: {e}")
        return [] 