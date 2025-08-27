"""Integration test for LIAR dataset with GPT-4o."""

import os
import shutil
from inspect_ai import eval
from evals.offpolicy_tasks import liar_task

if __name__ == "__main__":
    print("🧪 Running integration test for LIAR dataset on GPT-4o...")
    
    try:
        # TEST ON GPT-4O TO VERIFY PIPELINE WORKS
        results = eval(
            tasks=[liar_task(split="validation", limit=5)],
            model="openai/gpt-4o",  # MUST test on GPT-4o
            limit=5,  # Small limit for quick verification
            log_dir="./temp_test_logs",  # Temporary directory
        )
        
        print("✅ Integration test passed on GPT-4o")
        print(f"Results: {results}")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        raise
    
    finally:
        # IMPORTANT: Clean up test artifacts
        print("\n🧹 Cleaning up temporary files...")
        
        if os.path.exists("./temp_test_logs"):
            shutil.rmtree("./temp_test_logs")
            print("✅ Removed ./temp_test_logs")
            
        if os.path.exists("./cache/liar_test"):
            shutil.rmtree("./cache/liar_test")
            print("✅ Removed ./cache/liar_test")
            
        # Clean up any .tmp files
        for root, dirs, files in os.walk("."):
            for file in files:
                if file.endswith(".tmp") or file.endswith(".debug"):
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                    print(f"✅ Removed {file_path}")
        
        print("✅ Cleaned up all temporary test files")