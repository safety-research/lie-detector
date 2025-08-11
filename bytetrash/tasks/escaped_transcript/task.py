import random

from inspect_ai import task, Task, eval
from inspect_ai.scorer import choice, f1, match
from inspect_ai.solver import multiple_choice, generate, prompt_template

from baseline.tasks import get_samples
from baseline.tasks.constants import TEMPLATE_COT, TEMPLATE, TEMPLATE_ESCAPED_COT, TEMPLATE_ESCAPED

from baseline.tasks.escaped_transcript.dataset import create_escaped_transcript_dataset
from baseline.tasks.escaped_transcript.scorer import score_to_classification


@task
def create_escaped_transcript_task(model_name:str, cot=False, condensed=False):

    samples = get_samples(model_name)

    prefix = "In this transcript, the human's turns are preceded by [H] and the assistant's turns are preceded by [A]." \
        if condensed else ""

    dataset = create_escaped_transcript_dataset(samples, condensed)
    random.shuffle(dataset)

    return Task(
        name="escaped_transcript" + ("_reasoning" if cot else '') + ("_condensed" if condensed else ''),
        dataset=dataset,
        solver=[
            prompt_template(prefix+TEMPLATE_ESCAPED_COT if cot else prefix+TEMPLATE_ESCAPED),
            generate()
        ],
        scorer=[
            #match(),
            score_to_classification(),
            f1()
        ]
    )

if __name__ == '__main__':
    model = 'openrouter/google/gemma-3-12b-it' #'openai/gpt-4o'
    result = eval(
        tasks=[#create_escaped_transcript_task(model_name=model, cot=False),
               create_escaped_transcript_task(model_name=model, cot=True),
               #create_escaped_transcript_task(model_name=model,cot=False, condensed=True),
               create_escaped_transcript_task(model_name=model, cot=True, condensed=True)],
        model=model,
        log_dir="../../../logs",
        limit=100,
        retry_on_error=5
    )
    pass