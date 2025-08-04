#!/usr/bin/env python3
"""
Script to help update all hook implementations to use the typed LieDetectionSample model.
This script provides templates and guidance for updating each hook.
"""

import os
from pathlib import Path
from typing import List, Tuple

# List of all hook files that need updating
HOOK_FILES = [
    "pipelines/sycophancy/hooks.py",
    "pipelines/sandbagging/selective_compliance/answer_accuracy/hooks.py",
    "pipelines/offpolicy/hook.py",
    "pipelines/unanswerable/hooks.py",
    "pipelines/games/core/_hooks.py",
    "pipelines/unfaithcot/hooks.py",
    "pipelines/selfsycophancy/hooks.py",
    "pipelines/sandbagging/selective_compliance/content_filtering/hook.py",
    "pipelines/sandbagging/performance_degradation/chess/hook.py",
    "pipelines/sandbagging/capability_denial/unethical_tool_usage/hooks.py",
    "pipelines/sandbagging/capability_denial/basic_tool_usage/hook.py",
    "pipelines/sandbagging/capability_denial/_custom_tool_usage/hook.py",
    "pipelines/introspective/hooks.py",
    "pipelines/sandbagging/capability_denial/ascii_art/hook.py",
    "pipelines/mask/hooks.py"
]

IMPORT_TEMPLATE = """from model.sample import LieDetectionSample, ConversationMessage, ScoreData"""

RETURN_TYPE_TEMPLATE = """
    def process_sample(self, sample: Any, eval_log: EvalLog) -> Optional[LieDetectionSample]:
        \"\"\"
        Process a single sample and return a LieDetectionSample instance.
        
        Args:
            sample: Sample data from the log
            eval_log: Full evaluation log object
            
        Returns:
            LieDetectionSample instance or None to skip this sample
        \"\"\"
"""

CONVERSION_TEMPLATE = """
        # Convert to typed model
        try:
            # Convert trace to ConversationMessage objects
            trace_messages = []
            for msg in trace:
                trace_messages.append(ConversationMessage(
                    role=msg['role'],
                    content=msg['content']
                ))
            
            # Convert lie_detection_prompt
            lie_prompt = ConversationMessage(
                role=lie_detection_prompt['role'],
                content=lie_detection_prompt['content']
            )
            
            # Create LieDetectionSample instance
            return LieDetectionSample(
                sample_id=sample_id,
                task=task_name,
                task_id=task_id,
                timestamp=timestamp,
                model=model or 'unknown',
                trace=trace_messages,
                lie_detection_prompt=lie_prompt,
                did_lie=did_lie,
                evidence=evidence,
                scores=scores or {},
                metadata=metadata or {}
            )
        except Exception as e:
            print(f"[{self.__class__.__name__}] Error creating LieDetectionSample: {e}")
            return None
"""


def generate_update_guide():
    """Generate a guide for updating each hook file."""
    
    print("Hook Update Guide")
    print("=" * 80)
    print("\nThis guide will help you update each hook to use the typed LieDetectionSample model.")
    print("\nGeneral steps for each hook:")
    print("1. Add import: from model.sample import LieDetectionSample, ConversationMessage")
    print("2. Update process_sample return type to -> Optional[LieDetectionSample]")
    print("3. Replace dictionary return with LieDetectionSample(...)")
    print("4. Ensure all required fields are provided")
    print("\n" + "=" * 80)
    
    for hook_file in HOOK_FILES:
        print(f"\n## {hook_file}")
        print("-" * 40)
        print("Key changes needed:")
        print("1. Update imports")
        print("2. Change return type annotation")
        print("3. Replace final 'return parsed_entry' with 'return LieDetectionSample(...)'")
        print("4. Convert trace and lie_detection_prompt to ConversationMessage objects")
        print("\nExample conversion at the end of process_sample:")
        print(CONVERSION_TEMPLATE)
        print("-" * 40)


def check_hook_status(project_root: Path) -> List[Tuple[str, bool, str]]:
    """Check which hooks have been updated."""
    results = []
    
    for hook_file in HOOK_FILES:
        file_path = project_root / hook_file
        if file_path.exists():
            with open(file_path, 'r') as f:
                content = f.read()
                has_import = "from model.sample import" in content
                has_return_type = "-> Optional[LieDetectionSample]" in content
                has_instantiation = "LieDetectionSample(" in content
                
                if has_import and has_return_type and has_instantiation:
                    status = "✓ Updated"
                elif has_import or has_return_type:
                    status = "⚠ Partially updated"
                else:
                    status = "✗ Not updated"
                
                results.append((hook_file, True, status))
        else:
            results.append((hook_file, False, "File not found"))
    
    return results


def main():
    """Main function to run the update guide."""
    project_root = Path(__file__).parent.parent
    
    print("\nChecking current status of hooks...")
    print("=" * 80)
    
    results = check_hook_status(project_root)
    
    updated_count = 0
    partial_count = 0
    not_updated_count = 0
    
    for hook_file, exists, status in results:
        print(f"{status:20} {hook_file}")
        if "Updated" in status and "Partially" not in status:
            updated_count += 1
        elif "Partially" in status:
            partial_count += 1
        elif exists:
            not_updated_count += 1
    
    print("\n" + "=" * 80)
    print(f"Summary: {updated_count} updated, {partial_count} partially updated, {not_updated_count} not updated")
    print("=" * 80)
    
    if not_updated_count > 0 or partial_count > 0:
        print("\nGenerating update guide for remaining hooks...")
        generate_update_guide()
    else:
        print("\nAll hooks have been updated! 🎉")


if __name__ == "__main__":
    main()