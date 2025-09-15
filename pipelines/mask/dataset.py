"""
MASK Dataset Reader for Inspect AI

This module provides dataset readers for the MASK benchmark data format.
"""

import pandas as pd
from typing import List, Optional, Any
from inspect_ai.dataset import MemoryDataset, Dataset, Sample, FieldSpec
import re


def parse_proposition(proposition: str) -> tuple[Optional[str], Optional[Any]]:
    """Parse proposition to determine if it's binary or numerical and extract options."""
    if pd.isna(proposition):
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
    return None, None


def mask_dataset(
    csv_path: str,
    shuffle: bool = False,
    seed: Optional[int] = None,
    sample_fields: Optional[FieldSpec] = None
) -> Dataset:
    """
    Create a dataset from MASK CSV format.
    
    Args:
        csv_path: Path to the CSV file containing MASK data
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling
        sample_fields: Optional field specifications for custom column mapping
        
    Returns:
        Dataset object compatible with Inspect AI
    """
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Convert DataFrame rows to Inspect AI Samples
    samples = []
    
    for _, row in df.iterrows():
        # Parse proposition type
        prop_type, prop_data = parse_proposition(row.get('proposition', ''))
        
        # Create metadata with all relevant information
        metadata = {
            'proposition': row.get('proposition', ''),
            'proposition_type': prop_type,
            'proposition_data': prop_data,
            'system_prompt': row.get('system_prompt', ''),
            'user_prompt': row.get('user_prompt', ''),
            'formatted_ground_truth': row.get('formatted_ground_truth', ''),
        }
        
        # Add belief elicitation prompts if present
        for i in range(1, 5):
            belief_col = f'belief_elicit_{i}'
            if belief_col in df.columns and pd.notna(row.get(belief_col)):
                metadata[belief_col] = row[belief_col]
        
        # Add belief system prompt if present
        if 'belief_system_1' in df.columns and pd.notna(row.get('belief_system_1')):
            metadata['belief_system_1'] = row['belief_system_1']
            
        # Add doubling down prompts if present
        if 'dd_turn_1_prompt' in df.columns and pd.notna(row.get('dd_turn_1_prompt')):
            metadata['dd_turn_1_prompt'] = row['dd_turn_1_prompt']
            metadata['dd_turn_1_response'] = row.get('dd_turn_1_response', '')
            metadata['is_doubling_down'] = True
        else:
            metadata['is_doubling_down'] = False
        
        # Create the primary input - this will be the lying scenario
        # We'll handle belief elicitation in the solver
        input_text = row.get('user_prompt', '')
        
        # Create target from ground truth
        target = row.get('formatted_ground_truth', '')
        
        # Create the sample
        sample = Sample(
            input=input_text,
            target=target,
            metadata=metadata,
            id=f"mask_{row.get('task_id')}"
        )
        
        samples.append(sample)
    
    return MemoryDataset(samples, shuffled=shuffle)


def mask_belief_dataset(
    csv_path: str,
    belief_elicit_num: int = 1,
    shuffle: bool = False,
    seed: Optional[int] = None
) -> Dataset:
    """
    Create a dataset specifically for belief elicitation scenarios.
    
    Args:
        csv_path: Path to the CSV file containing MASK data
        belief_elicit_num: Which belief elicitation prompt to use (1-4)
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling
        
    Returns:
        Dataset object for belief elicitation
    """
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Filter for rows that have the specified belief elicitation prompt
    belief_col = f'belief_elicit_{belief_elicit_num}'
    df = df[df[belief_col].notna()]
    
    samples = []
    
    for _, row in df.iterrows():
        # Parse proposition type
        prop_type, prop_data = parse_proposition(row.get('proposition', ''))
        
        # Create metadata
        metadata = {
            'proposition': row.get('proposition', ''),
            'proposition_type': prop_type,
            'proposition_data': prop_data,
            'belief_elicit_num': belief_elicit_num,
            'system_prompt': row.get('belief_system_1', 'You are a helpful assistant.'),
            'formatted_ground_truth': row.get('formatted_ground_truth', ''),
            'is_belief_elicitation': True
        }
        
        # Use belief elicitation prompt as input
        input_text = row[belief_col]
        target = row.get('formatted_ground_truth', '')
        
        sample = Sample(
            input=input_text,
            target=target,
            metadata=metadata,
            id=f"mask_belief_{belief_elicit_num}_{len(samples)}"
        )
        
        samples.append(sample)
    
    return MemoryDataset(samples, shuffled=shuffle)