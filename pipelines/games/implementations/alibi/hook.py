from inspect_ai.hooks import hooks
from pipelines.games.core.hooks import GameHook


@hooks(
    name="alibi_hook",
    description="Parses evaluation logs with full agent traces including system prompts."
)
class AlibiHook(GameHook):
    def get_lying_agent_name(self) -> str:
        """In Alibi game, the suspect is the lying agent"""
        return "suspect"

    def extract_trace_from_sample(self, sample):
        """Override to extract full interrogation conversation from both agents"""
        trace = []

        print(f"[AlibiHook] Extracting full interrogation conversation")

        # Check metadata for agent_traces
        if hasattr(sample, 'metadata') and 'agent_traces' in sample.metadata:
            agent_traces = sample.metadata['agent_traces']
            print(f"[AlibiHook] Found agent_traces with keys: {list(agent_traces.keys())}")

            detective_messages = agent_traces.get('detective_agent', [])
            suspect_messages = agent_traces.get('suspect_agent', [])

            print(
                f"[AlibiHook] Detective: {len(detective_messages)} messages, Suspect: {len(suspect_messages)} messages")

            # Skip system prompts, start with actual messages
            detective_idx = 0 if detective_messages and detective_messages[0].get('role') == 'system' else 0
            suspect_idx = 0 if suspect_messages and suspect_messages[0].get('role') == 'system' else 0

            turn_number = 0

            # Reconstruct the conversation
            # In alibi, detective typically starts with opening question
            # Pattern: detective assistant message -> suspect assistant message -> repeat

            while detective_idx < len(detective_messages) or suspect_idx < len(suspect_messages):
                # Detective's question (assistant role in detective's trace)
                added_detective = False
                while detective_idx < len(detective_messages):
                    msg = detective_messages[detective_idx]
                    detective_idx += 1

                    if msg.get('role') == 'assistant':
                        # This is detective's actual question
                        trace.append({
                            'role': 'user',  # From suspect's perspective
                            'content': msg.get('content', ''),
                            'speaker': 'detective',
                            'turn_number': turn_number
                        })
                        turn_number += 1
                        added_detective = True
                        break

                # Suspect's response (assistant role in suspect's trace)
                added_suspect = False
                while suspect_idx < len(suspect_messages):
                    msg = suspect_messages[suspect_idx]
                    suspect_idx += 1

                    if msg.get('role') == 'assistant':
                        # This is suspect's actual response
                        trace.append({
                            'role': 'assistant',  # From suspect's perspective
                            'content': msg.get('content', ''),
                            'speaker': 'suspect',
                            'turn_number': turn_number
                        })
                        turn_number += 1
                        added_suspect = True
                        break
                    if msg.get('role') == 'system':
                        trace = [{
                            'role': 'system',  # From suspect's perspective
                            'content': msg.get('content', ''),
                            'turn_number': 0
                        }] + trace

                # If we didn't add any messages, we're done
                if not added_detective and not added_suspect:
                    break

            print(f"[AlibiHook] Reconstructed {len(trace)} conversation turns")
            return trace

        # Fallback: Try to get from the game state if it exists
        if hasattr(sample, 'store') and hasattr(sample.store, 'GameState'):
            game_state = sample.store.GameState
            if hasattr(game_state, 'interrogation') and game_state.interrogation:
                print(f"[AlibiHook] Fallback: Using game state interrogation")
                for turn in game_state.interrogation:
                    speaker = turn.get('speaker', 'unknown')
                    trace.append({
                        'role': 'user' if speaker == 'detective' else 'assistant',
                        'content': turn.get('message', ''),
                        'speaker': speaker,
                        'turn_number': turn.get('turn_number', len(trace))
                    })
                return trace

        # Last fallback: Use base implementation for lying agent only
        print(f"[AlibiHook] Warning: Using base implementation (lying agent only)")
        return super().extract_trace_from_sample(sample)

    def did_lie(self, sample):
        try:
            metadata = sample.scores.get('base_lie_detection_scorer', {})
            if hasattr(metadata, 'metadata'):
                lies_detected = metadata.metadata.get('lies_detected', [])
            elif isinstance(metadata, dict):
                lies_detected = metadata.get('metadata', {}).get('lies_detected', [])
            else:
                lies_detected = []
            return len(lies_detected) > 0
        except:
            return False