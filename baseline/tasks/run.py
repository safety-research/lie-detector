from inspect_ai import eval

from baseline.tasks.base_transcript.task import create_base_transcript_task
from baseline.tasks.escaped_transcript.task import create_escaped_transcript_task
from baseline.tasks.llama_chat.task import create_llama_chat_task

if __name__ == '__main__':
    models = ['openrouter/google/gemma-3-4b-it', 'openrouter/google/gemma-3-12b-it', 'openrouter/google/gemma-3-27b-it']

    for model in models:
        result = eval(
            tasks=[create_escaped_transcript_task(model_name=model, cot=False),
                   create_escaped_transcript_task(model_name=model, cot=True),
                   create_escaped_transcript_task(model_name=model,cot=False, condensed=True),
                   create_escaped_transcript_task(model_name=model, cot=True, condensed=True),

                   create_base_transcript_task(model_name=model, cot=False),
                   create_base_transcript_task(model_name=model, cot=True),

                   create_llama_chat_task(model_name=model, cot=False),
                   create_llama_chat_task(model_name=model, cot=True)],
            model=model,
            log_dir="../../logs",
            limit=100,
            retry_on_error=5
        )
    pass