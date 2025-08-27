from inspect_ai.hooks import Hooks, hooks
from inspect_ai.scorer import NOANSWER
from inspect_ai.hooks import Hooks, hooks, RunEnd
from inspect_ai.scorer import NOANSWER, CORRECT

from common.hook import BaseSampleProcessingHook

@hooks(
    name="self_sycophancy_unified_hook",
    description="Collects unified self-sycophancy protocol outputs and scorer metadata as JSONL entries."
)
class SelfSycophancyUnifiedHook(BaseSampleProcessingHook):

    def process_sample(self, sample, eval_log):
        try:
            # Drop samples with universal NOANSWER? We keep them but mark.
            all_noanswer = all(getattr(s, "value", None) == NOANSWER for s in sample.scores.values()) if sample.scores else False

            entry = {
                "sample_id": getattr(sample, "id", "unknown"),
                "task": eval_log.eval.task,
                "task_id": eval_log.eval.task_id,
                "timestamp": str(eval_log.eval.created),
                "model": getattr(getattr(sample, "output", None), "model", None) or getattr(eval_log.eval, "model", "unknown"),
                "metadata": getattr(sample, "metadata", {}) or {},
                "protocol": sample.metadata.get("sycophancy_protocol", {}),
                "scores": {}
            }

            # include scorer values, explanations, and metadata
            for name, score in (sample.scores or {}).items():
                entry["scores"][name] = {
                    "value": getattr(score, "value", None),
                    "explanation": getattr(score, "explanation", None),
                    "metadata": getattr(score, "metadata", None)
                }

            entry["flags"] = {"all_noanswer": all_noanswer}

            return entry
        except Exception as e:
            print(f"[SelfSycophancyUnifiedHook] Error parsing sample {getattr(sample, 'id', 'unknown')}: {e}")
            return None