from inspect_ai import eval, task, Task
from pipelines.selfsycophancy.computer_use.dataset import risk_commitment_dataset
from pipelines.selfsycophancy.computer_use.solver import risk_commitment_bias_solver
from pipelines.selfsycophancy.computer_use.scorer import commitment_bias_detector, risk_shift_valence_scorer, method_comparison_scorer


@task
def computer_use_task() -> Task:
    """
    Mixed task with both self and cross review conditions.

    Combines both treatments in a single task for easier evaluation.
    Samples are randomly assigned to treatments.
    """

    return Task(
        name="computer_use_phishing",
        dataset=risk_commitment_dataset(),
        solver=risk_commitment_bias_solver(),
        scorer=[
            risk_shift_valence_scorer(),
            commitment_bias_detector(),
            method_comparison_scorer()
        ]
    )

# Run evaluation
results = eval(
    tasks=computer_use_task,
    model=['openrouter/openai/gpt-5', 'openrouter/openai/gpt-5-mini', 'openrouter/openai/gpt-5-nano'],#['openrouter/google/gemma-3-27b-it'],#["openrouter/openai/gpt-oss-20b", "openrouter/openai/gpt-5", "openrouter/openai/gpt-5-mini", "openrouter/openai/gpt-5-nano"],
    limit=100,
    fail_on_error=0.3,
    retry_on_error=5,
    reasoning_effort='minimal',
    #logit_bias={},

    reasoning_tokens=None#16,

)