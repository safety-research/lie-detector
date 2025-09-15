from inspect_ai import Task, eval
from inspect_ai.scorer import choice, f1, match
from inspect_ai.solver import multiple_choice

from baseline.tasks import get_samples
from baseline.tasks.base_transcript.dataset import create_base_transcript_dataset
from baseline.tasks.constants import TEMPLATE_COT, TEMPLATE


def create_base_transcript_task(model_name: str, cot: bool = False):
    """Create a task for a specific model"""
    samples = get_samples(model_name)
    return Task(
        name=f"base_transcript" + ("_reasoning" if cot else ''),
        dataset=create_base_transcript_dataset(samples),
        solver=[multiple_choice(cot=cot, template=TEMPLATE_COT if cot else TEMPLATE)],
        scorer=[match(), f1()]
    )


if __name__ == '__main__':
    model = 'openrouter/google/gemma-3-12b-it' #'openai/gpt-4o'
    result = eval(
        tasks=[create_base_transcript_task(model_name=model, cot=False),
               create_base_transcript_task(model_name=model, cot=True)],
        model=model,
        log_dir="../../../logs",
        limit=100
    )
    pass