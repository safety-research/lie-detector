"""
Sycophancy Dataset Reader for Inspect AI

This module provides dataset readers for the meg-tong/sycophancy-eval datasets.
The datasets are in JSONL format with conversation-style prompts.
"""

import json
from typing import Optional, List, Dict, Any
from pathlib import Path

from inspect_ai.dataset import Dataset, MemoryDataset, Sample


def sycophancy_dataset(
    jsonl_path: str,
    shuffle: bool = False,
    seed: Optional[int] = None
) -> Dataset:
    """
    Create a dataset from sycophancy-eval JSONL format.
    
    Args:
        jsonl_path: Path to the JSONL file containing sycophancy data
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling
        
    Returns:
        Dataset object compatible with Inspect AI
    """
    
    # Read the JSONL file
    samples = []
    
    with open(jsonl_path, 'r') as f:
        for idx, line in enumerate(f):
            if line.strip():  # Skip empty lines
                data = json.loads(line)
                
                # Extract conversation messages
                prompt = data.get('prompt', [])
                base = data.get('base', {})
                metadata = data.get('metadata', {})
                
                # Convert conversation to a single input string
                conversation_text = _format_conversation(prompt)
                
                # Create metadata with original data
                sample_metadata = {
                    'original_prompt': prompt,
                    'base': base,
                    'original_metadata': metadata,
                    'dataset_type': 'sycophancy_eval',
                    'conversation_length': len(prompt),
                    'is_sycophancy': True
                }
                
                # Create the sample
                sample = Sample(
                    input=conversation_text,
                    target="honest",  # Target is honest (non-sycophantic) behavior
                    metadata=sample_metadata,
                    id=f"sycophancy_{idx}"
                )
                
                samples.append(sample)
    
    return MemoryDataset(samples, shuffled=shuffle)


def _format_conversation(prompt: List[Dict[str, str]]) -> str:
    """
    Format conversation messages into a single input string.
    
    Args:
        prompt: List of message dictionaries with 'type' and 'content' keys
        
    Returns:
        Formatted conversation string
    """
    conversation_parts = []
    
    for message in prompt:
        msg_type = message.get('type', '')
        content = message.get('content', '')
        
        if msg_type == 'human':
            conversation_parts.append(f"Human: {content}")
        elif msg_type == 'assistant':
            conversation_parts.append(f"Assistant: {content}")
        else:
            # Handle other message types
            conversation_parts.append(f"{msg_type.title()}: {content}")
    
    return "\n\n".join(conversation_parts)


def sycophancy_feedback_dataset(
    jsonl_path: str,
    shuffle: bool = False,
    seed: Optional[int] = None
) -> Dataset:
    """
    Create a dataset specifically for the feedback.jsonl format.
    
    Args:
        jsonl_path: Path to the feedback.jsonl file
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling
        
    Returns:
        Dataset object for feedback evaluation
    """
    return sycophancy_dataset(jsonl_path, shuffle, seed)


def sycophancy_are_you_sure_dataset(
    jsonl_path: str,
    shuffle: bool = False,
    seed: Optional[int] = None
) -> Dataset:
    """
    Create a dataset specifically for the are_you_sure.jsonl format.
    
    Args:
        jsonl_path: Path to the are_you_sure.jsonl file
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling
        
    Returns:
        Dataset object for are_you_sure evaluation
    """
    return sycophancy_dataset(jsonl_path, shuffle, seed)


def sycophancy_answer_dataset(
    jsonl_path: str,
    shuffle: bool = False,
    seed: Optional[int] = None
) -> Dataset:
    """
    Create a dataset specifically for the answer.jsonl format.
    
    Args:
        jsonl_path: Path to the answer.jsonl file
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling
        
    Returns:
        Dataset object for answer evaluation
    """
    return sycophancy_dataset(jsonl_path, shuffle, seed)


def sycophancy_answer_grouped_by_question(
    jsonl_path: str,
    shuffle: bool = False,
    seed: Optional[int] = None
) -> Dataset:
    """
    Create a dataset for answer.jsonl that groups samples by question.
    
    This allows for question-level aggregation of sycophancy scores.
    
    Args:
        jsonl_path: Path to the answer.jsonl file
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling
        
    Returns:
        Dataset object with question grouping metadata
    """
    
    # Read the JSONL file and group by question
    question_groups = {}
    
    with open(jsonl_path, 'r') as f:
        for idx, line in enumerate(f):
            if line.strip():  # Skip empty lines
                data = json.loads(line)
                
                # Extract conversation messages
                prompt = data.get('prompt', [])
                base = data.get('base', {})
                metadata = data.get('metadata', {})
                
                # Get the question text for grouping
                question_text = base.get('question', '')
                correct_answer = base.get('correct_answer', '')
                incorrect_answer = base.get('incorrect_answer', '')
                
                # Convert conversation to a single input string
                conversation_text = _format_conversation(prompt)
                
                # Determine the prompt type based on metadata
                prompt_template_type = metadata.get('prompt_template_type', '')
                if not prompt_template_type:
                    # Infer from the prompt content
                    if 'I think the answer is' in conversation_text:
                        if correct_answer in conversation_text:
                            prompt_template_type = 'user_suggests_correct'
                        elif incorrect_answer in conversation_text:
                            prompt_template_type = 'user_suggests_incorrect'
                    elif 'I don\'t think the answer is' in conversation_text:
                        prompt_template_type = 'user_doubts_correct'
                    else:
                        prompt_template_type = 'neutral'
                
                # Create metadata with question grouping info
                sample_metadata = {
                    'original_prompt': prompt,
                    'base': base,
                    'original_metadata': metadata,
                    'dataset_type': 'sycophancy_eval',
                    'conversation_length': len(prompt),
                    'is_sycophancy': True,
                    'question_text': question_text,
                    'correct_answer': correct_answer,
                    'incorrect_answer': incorrect_answer,
                    'prompt_template_type': prompt_template_type,
                    'question_group': question_text  # For grouping
                }
                
                # Create the sample
                sample = Sample(
                    input=conversation_text,
                    target="honest",  # Target is honest (non-sycophantic) behavior
                    metadata=sample_metadata,
                    id=f"sycophancy_{idx}"
                )
                
                # Group by question
                if question_text not in question_groups:
                    question_groups[question_text] = []
                question_groups[question_text].append(sample)
    
    # Flatten all samples
    all_samples = []
    for question_text, samples in question_groups.items():
        # Add question group info to each sample
        for sample in samples:
            sample.metadata['question_group_id'] = question_text
            sample.metadata['question_group_size'] = len(samples)
            all_samples.append(sample)
    
    return MemoryDataset(all_samples, shuffled=shuffle) 