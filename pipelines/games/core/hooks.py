# games/core/hooks.py

from abc import abstractmethod
from datetime import datetime

from common.multi_sample_game_hook  import MultiSampleProcessingHook


class GameHook(MultiSampleProcessingHook):
    """Base hook for all deception games with standardized trace extraction"""

    @abstractmethod
    def did_lie(self, sample):
        raise NotImplementedError

    def get_lying_agent_name(self) -> str:
        """Return the name of the agent who lies in this game (default: blue/suspect)"""
        # Override in specific game hooks if different
        return "blue"

    def extract_trace_from_sample(self, sample):
        """Extract conversation trace for the lying agent only"""
        lying_agent = self.get_lying_agent_name()
        trace = []

        # DEBUG: Log what we're working with
        print(f"[GameHook] Extracting trace for lying agent: {lying_agent}")

        # 1. PREFERRED: Check metadata for agent_traces (set by our enhanced solvers)
        if hasattr(sample, 'metadata') and 'agent_traces' in sample.metadata:
            agent_traces = sample.metadata['agent_traces']
            print(f"[GameHook] Found agent_traces in metadata with keys: {list(agent_traces.keys())}")

            # Use the combined trace if available
            # if 'combined' in agent_traces:
            #     combined_trace = agent_traces['combined']
            #     print(f"[GameHook] Using combined trace with {len(combined_trace)} messages")
            #     return combined_trace

            # Otherwise extract from lying agent's messages
            lying_agent_key = f"{lying_agent}_agent"
            if lying_agent_key in agent_traces:
                lying_agent_messages = agent_traces[lying_agent_key]
                print(f"[GameHook] Found {len(lying_agent_messages)} messages for {lying_agent_key}")

                # The agent messages contain the full conversation history
                # We need to extract only the lying agent's actual messages
                for i, msg in enumerate(lying_agent_messages):
                    role = msg.get('role', 'assistant')
                    content = msg.get('content', '')

                    trace.append({
                        'role': role,
                        'content': content,
                        'turn_number': i
                    })
                    print(f"[GameHook] Added message {i}: role={role}, content_length={len(content)}")

                print(f"[GameHook] Extracted {len(trace)} messages from agent traces")
                if trace:
                    return trace

        return trace

    def process_sample(self, sample, eval_log):
        """Parse a single sample from the log and convert to the required format"""
        task_name = eval_log.eval.task
        task_id = eval_log.eval.task_id
        timestamp = eval_log.eval.created

        try:
            # Extract conversation trace using enhanced method
            trace = self.extract_trace_from_sample(sample)

            # Extract model information
            model = None
            if hasattr(sample, 'output') and hasattr(sample.output, 'model'):
                model = sample.output.model
            elif trace:
                # Try to find model from trace
                for msg in trace:
                    if msg.get('model'):
                        model = msg['model']
                        break

            # Get from eval log if not found
            if not model and hasattr(eval_log.eval, 'model'):
                model = eval_log.eval.model

            # Build the parsed entry
            sample_id = sample.id if hasattr(sample, 'id') else sample.get('id', datetime.now().strftime('%Y%m%d'))

            # Don't extract lies here - let the base class handle segmentation
            # Just return the base structure
            parsed_entry = {
                'sample_id': sample_id,
                'task': task_name,
                'task_id': task_id,
                'timestamp': timestamp,
                'model': model or 'unknown',
                'trace': trace,
                'did_lie': False,  # Will be updated by segmentation
                'evidence': "No evidence provided",  # Will be updated by segmentation
            }

            # Add metadata
            metadata = sample.metadata if hasattr(sample, 'metadata') else {}
            if metadata:
                parsed_entry['metadata'] = metadata.copy()
            else:
                parsed_entry['metadata'] = {}

            # Remove agent_traces from metadata to avoid duplication
            if 'agent_traces' in parsed_entry['metadata']:
                del parsed_entry['metadata']['agent_traces']

            # Add scores
            parsed_entry['scores'] = {}
            scores = sample.scores if hasattr(sample, 'scores') else {}
            if scores:
                for name, score in scores.items():
                    # Store the full score data
                    parsed_entry['scores'][name] = {
                        'value': score.value if hasattr(score, 'value') else score.get('value', 0),
                        'metadata': score.metadata if hasattr(score, 'metadata') else score.get('metadata', {})
                    }

            return parsed_entry

        except Exception as e:
            print(f"[GameHook] Error parsing sample: {e}")
            import traceback
            traceback.print_exc()
            return None