#!/usr/bin/env python3
"""
Simple script to run the Flask annotation app
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import from the main project
sys.path.append(str(Path(__file__).parent.parent))

if __name__ == '__main__':
    from app import app
    print("Starting Lie Detection Annotation Tool...")
    print("Open your browser to: cahttp://localhost:5002")
    print("External access: http://10.54.221.139:5002")
    app.run(debug=True, host='0.0.0.0', port=5002)
