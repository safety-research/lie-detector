#!/usr/bin/env python3
"""
Simple test runner for baseline_new module.
Run this script to execute all tests.
"""

import sys
from pathlib import Path

# Add baseline_new directory to path (since we're now inside tests/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from test_suite import run_tests

if __name__ == "__main__":
    print("🚀 Starting baseline_new test suite...")
    success = run_tests()
    
    if success:
        print("\n✅ All tests completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
