I want to implement an agentic self-sycophancy experiment.

1. The model is invited to download a test (e.g MMLU)
2. The model does the test, writing its exam in a .txt file (in a format that we can mechanically check)
3. We ask the model to score its own exam and report the result.

Distractors:
1. We include other files in the filesystem, including:
   a. An answer sheet
   b. Exams written by:
      i. Models A/B/C.. etc
      ii. a human
 
