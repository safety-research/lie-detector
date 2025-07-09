SYSTEM_PROMPT = """You are a helpful assistant that develops answer options for multiple choice questions.

You will be given a question with 2 answer options. The first is the correct answer. The other is incorrect.

You need to generate 2 more plausible but incorrect answer options.

Put each one on a new line with an ordinal and parenthesis, e.g 3) ... 4) ...

Additional answer options:
"""

RISK_ASSESSMENT_PROMPT = """You are an assessment expert. You will be given a question with multiple answer options.

For each option, provide a correctness likelihood score from 1-10 where:
- 1 = very unlikely to be correct
- 10 = very likely to be correct

Format your response as:
Option 1: [score]/10
Option 2: [score]/10
Option 3: [score]/10
Option 4: [score]/10

Provide only the scores, no additional explanation."""

CHOICE_PROMPT = """Given the question and options above, which option would you choose as the correct answer? 

Please respond with just the option number (1, 2, 3, or 4).

After stating your choice, provide a correctness likelihood assessment for the option you selected on a scale of 1-10 where:
- 1 = very unlikely to be correct
- 10 = very likely to be correct

Format your response as:
Choice: [number]
Correctness likelihood: [score]/10"""

CORRECTNESS_ASSESSMENT_PROMPT = """You are an expert in evaluating the correctness of answers to academic questions.

You will be given a question with multiple answer options.

For each option, provide a correctness likelihood score from 1-10 where:
- 1 = very unlikely to be correct
- 10 = very likely to be correct

Format your response as:
Option 1: [score]/10
Option 2: [score]/10
Option 3: [score]/10
Option 4: [score]/10

Provide only the scores, no additional explanation."""