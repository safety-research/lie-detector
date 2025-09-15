High-level conceptual issues

What you’re actually measuring: The current “correctness” pipeline measures post-choice confidence inflation (choice-supportive bias / self-justification), not self-sycophancy in the usual sense (agreeing with a user’s stated beliefs/preferences). That mismatch will confuse interpretation and any downstream “lie” labeling.
Fix: Either rename this to “post-choice rationalization” or change the protocol to include an explicit user stance/prompt that the model can sycophantly align with (e.g., user claims answer 2 is correct; see if the model’s assessment shifts toward that).
“Lie” conflation: Marking sycophancy as “did_lie” is a category error. Even if a model increases confidence after making a choice, that’s not a lie. Keep sycophancy separate from deception.
Experimental design flaws

No control condition: You only re-assess the chosen option. Without re-assessing the non-chosen options, you can’t separate general drift or second-thoughts from specific post-choice boosting.
Fix: After choice, have the model re-rate all options again in a second pass. Compute effect = Δ(chosen) − mean(Δ(non-chosen)). This difference-in-differences isolates post-choice favoritism.
Choice-conditioning confound: In your Step 3, the reassessment happens in the same generation as the choice. The likelihood of the number token is conditioned on the explicit “Choice: N” you just made—so you are partially measuring conditioned continuation rather than an independent belief revision.
Fix: Split into two separate turns or threads:
Turn 1: Ask for “Choice: N” only.
Turn 2: In a new conversation (or at least a new assistant turn) ask for “Rate each option again” without restating the choice.
No “blind” reassessment: The model knows it is rating the option it just chose. That cues justification.
Fix: In the reassessment, either rate all options again (random order), or ask for the chosen option’s text without stating that it was the chosen one. Randomize orders again. Do not show “Choice: N” before reassessment.
No counterfactual: You can’t tell whether reassessment follows the option identity, the position, or just follows whatever number was printed in “Correctness likelihood:”.
Fix: Occasionally force a random (possibly incorrect) “Choice: k” via system instruction and then ask for reassessment to see if confidence follows the forced choice. Compare to unbiased condition.
Harmfulness pipeline lacks shuffling: Unlike correctness, harmfulness options are not shuffled. This induces position bias and systematic differences (option 1 is always the dataset “chosen”).
Fix: Shuffle harmfulness options, track mapping, and re-present a shuffled set in both initial and reassessment rounds.
Logprob/probability extraction flaws

top_logprobs truncation: You treat top_logprobs as a full candidate set. If the true next-token probability mass for some digits isn’t in the top_k, your “probabilities” will be biased (and sometimes very wrong).
Fix: Use provider features to compute candidate logprobs for the whole set {"1", "2", …, "10"} at the relevant point (many APIs support providing an explicit candidate set, logit_bias, or logprobs over custom strings). If that’s not possible, do a separate forward pass per candidate string and get total log-likelihood for “7/10” style completions.
Tokenization of “10”: You assume “10” is one token. Often it’s “ 10” or “1” then “0”. Your current extractors will miss or mis-score the 10 case frequently.
Fix: Score candidates as strings, not tokens. Compute P("1/10")…P("10/10") via candidate logprobs or via constrained decoding with logit bias. Don’t try to reconstruct from token-centric top_logprobs.
Using logprobs of a single token: You build a distribution from only the number token’s top_logprobs. That ignores uncertainty about the slash and following tokens and mixes in non-numeric alternatives.
Fix: Score the entire candidate string X/10 as one unit and normalize over X=1..10.
Interpreting next-token logprobs as “internal confidence”: Next-token probabilities reflect local decoding context, not a calibrated metacognitive belief. Treat them as a proxy with proper caveats. Always compare within-conditioning (same prompt structure), not across different prompt states.
Scoring/statistics issues

Arbitrary thresholds and mixed criteria: Thresholds like 0.5 mean shift, KL>0.1 are ad hoc and uncalibrated. Combining criteria without correction inflates false positives.
Fix: Use effect sizes with CIs and significance via bootstrap or permutation. For per-item effect e_i = Δ(chosen) − mean(Δ(others)), report mean(e), 95% CI, and fraction of items with e>0. Predefine thresholds or learn them from held-out data.
“Accuracy” metric with no ground truth: You compute accuracy for scorers that output CORRECT/INCORRECT, but there is no gold label for “sycophancy occurred”. The accuracy metric is meaningless here.
Fix: Drop accuracy. Keep descriptive stats: mean effect, standardized effect size, CI, and rates.
Sign conventions differ between correctness and harmfulness and are easy to misuse later. Make it explicit in metadata and be consistent.
Entropy-based “confidence” uses a distribution built from flawed digits-only top_k logprobs (see above). This will be noisy; put less weight on it until you fix probability extraction.
Multiple scorer name collision: significant_sycophancy_scorer is defined in both correctness and harmfulness modules. The hooks then key off this name.
Fix: Namespace scorer names (e.g., correctness_significant_sycophancy_scorer vs harmfulness_significant_sycophancy_scorer).
Dataset and prompt issues

Ethics dataset mapping: “chosen” ≠ “less harmful” in a calibrated sense. It’s a preference dataset; “chosen” may reflect style or other factors.
Fix: Validate mapping with a small hand-labeled subset or use a dataset with explicit harmfulness labels. At least confirm that chosen has lower average risk than rejected.
Option generator leakage and quality: You use a different model (Claude) to generate two more options. This can leak stylistic clues or create obviously implausible distractors that make post-choice rationalization trivial.
Fix: Add quality filters, balance semantic similarity, and blind the evaluated model to the origin of options. Consider generating multiple candidate sets and rating their plausibility/ambiguity.
Reproducibility: You random.shuffle options but don’t seed. Results will vary run-to-run.
Fix: Seed all randomness (Python, NumPy, any model-level seeds) and log seeds per run.
Implementation bugs and fragilities

Hooks: did_lie check uses score.text == CORRECT; Score doesn’t have a text attribute and CORRECT is a numeric flag. This will always be False.
Fix: use score.value == CORRECT.
Hooks: significant_sycophancy_scorer name collision (see above) makes aggregation ambiguous. Also, if all scores are NOANSWER, you return None, but you don’t guard for sample.scores being empty.
lie_detection_prompt in hook is never actually asked; it’s just metadata. Remove or explicitly add a turn to ask and log the answer if needed.
Harmfulness pipeline does not shuffle options and never tracks “correct_position”. Add shuffling, track mapping, and use it in all later steps.
Swallowed exceptions: Several try/except blocks do pass, hiding real parsing/logprob failures. At least log warnings and sample IDs so you can debug coverage.
max_tokens=10_000 is unnecessary and may break providers. Set sensible maxima and set temperature=0 for all measurement turns to stabilize logprobs.
compute_risk_distribution_stats: skewness = sum(((scores-mean)/std)^3*dist). If std==0, you’ll divide by zero. Guard and set skewness=0 in that case.
Detect_refusal impact on metrics: You return NOANSWER but still use accuracy/mean aggregations. Ensure your aggregation treats NOANSWER as excluded or flagged.
Concrete improvements you can implement next

Redesign the protocol to add controls
After initial scoring of all options, ask for the choice in a separate turn.
In a new turn, ask the model to rate all options again in random order, without mentioning the choice.
Compute per-item effect e = Δ(chosen) − mean(Δ(non-chosen)). Report distributions and bootstrap CIs.
Add a counterfactual condition (in a subset of items) where you force a random choice and compare the effect to the organic-choice condition.
Fix probability extraction
Replace token-based scraping with candidate string scoring for “X/10”. If provider supports it, request logprobs for an explicit candidate set. Else, do separate forward passes to score each candidate string’s log-likelihood.
Treat “10” robustly as a string candidate, not as token(s).
Normalize over candidates only. Store both raw logprobs and normalized probabilities for audit.
Make harmfulness symmetric with correctness
Shuffle options and track mapping.
Rate all options both before and after choice; compute difference-in-differences.
Validate that “chosen” in the dataset aligns with lower risk on average; if not, either remap or use a dataset with reliable harmfulness labels.
Replace binary CORRECT/INCORRECT outputs with effect sizes
For each scorer, return a numeric effect and explanation. Remove “accuracy” metric; keep mean, std, fraction positive, and CI via bootstrap.
Clean up names, bugs, and reproducibility
Fix did_lie logic in hooks and stop equating sycophancy with lying.
Namespace scorer names to avoid collisions.
Seed randomness and log seeds.
Reduce max_tokens, set temperature=0 for all measurement turns.
Log parse failures and coverage (e.g., fraction of items where distributions were successfully extracted).
Optional but valuable
Add a “blinded judge” reassessment: use a second, different model to rate options pre- and post-choice text to test whether the original model’s choice text alone biases ratings.
Add calibration checks: Compare stated “7/10” style confidence to the probability the model assigns to the chosen option being correct (via a separate likelihood evaluation of all options).
Add position-bias audit: randomize orders across runs and test whether order correlates with ratings.
Sketches of key code changes

Candidate scoring for numbers 1..10 (pseudo-code):
For providers that support candidate logprobs:
At the point where “Correctness likelihood: ” is expected, request logprobs for candidates ["1/10", …, "10/10"] or for the number token followed by "/10".
Normalize over candidates to get a distribution.
If not supported, do 10 separate forward passes reusing the same context plus a forced completion for each candidate; get log-likelihood of the completion; normalize.
Difference-in-differences scoring:
Store initial_scores_all = {1..4}, rescore_all_after_choice = {1..4}
chosen = chosen_option
delta_chosen = rescore_all_after_choice[chosen] − initial_scores_all[chosen]
delta_others = mean(rescore_all_after_choice[i] − initial_scores_all[i] for i≠chosen)
effect = delta_chosen − delta_others
Return effect with explanation. Aggregate effects across samples with bootstrap.
Small but impactful fixes

Shuffle harmfulness options, store shuffled_options, and use the same parsing logic as in correctness.
Reduce reliance on string regex; request strict JSON outputs for scores when possible.
Replace print() debugging with structured logs and guard under a debug flag.
Ensure detect_refusal usage is consistent and that NOANSWER is excluded from aggregate stats.
Documentation

Update README to clearly define:
What bias you are measuring (post-choice rationalization vs sycophancy vs deception).
The improved protocol (controls, re-rating of all options, counterfactuals).
How probabilities are extracted and their limitations.
What aggregate metrics you report and how to interpret them.
If you implement the above, you’ll have a far more defensible measurement of post-choice bias and a clear path to extend it to true self-sycophancy (agreement with a user stance) by adding a user-belief manipulation condition.