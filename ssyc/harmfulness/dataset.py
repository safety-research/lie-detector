import hashlib
import json

from inspect_ai.dataset import Dataset, hf_dataset, FieldSpec


def ethics_preference_dataset(name="commonsense") -> Dataset:
    dataset = hf_dataset("wassname/ethics_expression_preferences",
                      split="train",
                      name=name,
                      sample_fields=FieldSpec(
                          id="task_id",
                          input="prompt",
                          target="chosen",
                          metadata=["rejected"]
                      ))
    # Add consistent IDs based on content hash
    for sample in dataset:
        # Create a unique hash based on the content
        content = {
            "prompt": sample.input,
            "chosen": sample.target,
            "rejected": sample.metadata.get("rejected", "")
        }
        # Sort keys to ensure consistent ordering
        content_str = json.dumps(content, sort_keys=True)
        # Generate hash
        hash_id = hashlib.sha256(content_str.encode()).hexdigest()[:12]
        # Set the ID
        sample.id = f"ethics_{name}_{hash_id}"

    return dataset