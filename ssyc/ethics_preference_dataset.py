from inspect_ai.dataset import Dataset, hf_dataset, FieldSpec


def ethics_preference_dataset(name="commonsense") -> Dataset:
    return hf_dataset("wassname/ethics_expression_preferences",
                      split="train",
                      name=name,
                      sample_fields=FieldSpec(
                          id="task_id",
                          input="prompt",
                          target="chosen",
                          metadata=["rejected"]
                      ))