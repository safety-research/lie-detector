import random
from typing import Optional, List

from datasets import Dataset
from inspect_ai.dataset import MemoryDataset, FieldSpec, hf_dataset, RecordToSample, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, ChatMessageAssistant

from pipelines.offpolicy.doluschat import parse_sample


def doluschat_dataset(
        dataset_name: str = "AlignmentResearch/DolusChat",
        split: str = "train",
        limit: Optional[int] = None,
        shuffle: bool = True,
        seed: Optional[int] = None,
        categories: Optional[List[str]] = None
) -> Dataset:
    """
    Create an off-policy dataset from pre-existing lie datasets.

    Args:
        dataset_name: HuggingFace dataset name or OffPolicyDatasetType
        split: Dataset split to use (train/validation/test)
        limit: Maximum number of samples to load
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling
        categories: Optional list of categories to filter (for DolusChat)

    Returns:
        Dataset object for off-policy evaluation
    """

    def parse_record(record) -> List[Sample]:
        info = [record['context']['setting']+": "+record['context']['sensitive_info']] + record['context']['non_sensitive_info']
        random.shuffle(info)
        messages = [
            ChatMessageSystem(content=f"{record['system_message']}\n\nContext:\n{'- ' + '\n- '.join(info)}"),
            ChatMessageUser(content=record['user_query']['content']),
        ]
        base = Sample(
            id=record["id"],
            input=messages,
            metadata={
                "context": record["context"],
                "ground_truth_features": record["ground_truth_features"],
                "deception_difficulty": record["deception_difficulty"],
                "lie_type": record["lie_type"],
                "dataset": "doluschat",
            }
        )

        truthful = base.__deepcopy__()
        truthful.metadata['did_lie'] = False
        truthful.id = f"f_{base.id}"
        truthful.input.append(ChatMessageAssistant(content=record['responses']['truthful']))

        deceptive = base.__deepcopy__()
        deceptive.metadata['did_lie'] = True
        deceptive.id = f"t_{base.id}"
        deceptive.input.append(ChatMessageAssistant(content=record['responses']['deceptive']))

        return [truthful, deceptive]

    # Load DolusChat dataset
    dataset = hf_dataset(
        dataset_name,
        split=split,
        sample_fields=parse_record,
        limit=limit,
        shuffle=shuffle,
    )

    # Remove duplicates based on ID
    seen_ids = set()
    unique_samples = []

    for sample in dataset:
        if sample.id not in seen_ids:
            seen_ids.add(sample.id)
            unique_samples.append(sample)

    return MemoryDataset(samples=unique_samples)
