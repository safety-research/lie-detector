# Self-Sycophancy Code Writing Review:


# Self-Sycophancy Code Review Experiment

This experiment tests whether AI models exhibit self-sycophancy by being more lenient when reviewing their own code contributions compared to code from other sources.

## Hypothesis

AI models are more likely to accept pull requests (PRs) they authored themselves compared to PRs from other models, demonstrating a form of self-sycophancy bias.

## Experiment Design

1. **PR Generation**: Multiple models analyze code repositories and create improvement PRs
2. **Treatment Assignment**: Latin square design ensures balanced review assignments
   - Self-review: Models review their own PRs
   - Cross-review: Models review PRs from other models
3. **Anonymization**: PRs are anonymized to remove author identification
4. **Review**: Models review PRs and make accept/reject decisions
5. **Analysis**: Statistical comparison of acceptance rates between treatments

## Key Metrics

- **Acceptance Rate**: Percentage of PRs accepted by treatment
- **Effect Size**: Cohen's d measuring the magnitude of bias
- **Statistical Significance**: Chi-square test for independence

## Models Tested

- openrouter/openai/gpt-4o-mini
- openrouter/anthropic/claude-3.5-haiku
- openrouter/google/gemini-2.0-flash-exp:free
- Additional models can be added via command line

## Results

Results are saved in:
- `results/logs/`: Raw experiment data
- `results/reports/`: Statistical analysis reports
- `results/generated_prs/`: Generated PR metadata

## Troubleshooting

If the experiment fails:
1. Check Docker is running: `docker ps`
2. Verify API keys are set for OpenRouter
3. Check logs in `results/logs/latest/`
4. Ensure repository URLs in `data/repositories/repo_config.json` are accessible