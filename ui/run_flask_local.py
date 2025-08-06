#!/usr/bin/env python3
"""
Simple Flask App Runner for Lie Detection Data Viewer

This script runs only the Flask web app using local data.
No S3 syncing - just reads from the local data directory.
"""

import os
import sys

# Add the current directory to Python path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set environment variables for the Flask app
os.environ['USE_LOCAL_DATA'] = 'True'
os.environ['LOCAL_DATA_PATH'] = '/home/ec2-user/s3_data'

def run_flask_app():
    """Run the Flask web application"""
    print("Starting Flask web application...")
    print(f"Using local data path: {os.environ.get('LOCAL_DATA_PATH')}")
    print(f"USE_LOCAL_DATA: {os.environ.get('USE_LOCAL_DATA')} (reading from local filesystem)")
    print("No S3 syncing - using existing local data only")
    print("")
    try:
        # Import and run the Flask app
        from flask_app import app
        app.run(debug=False, host='0.0.0.0', port=8080, use_reloader=False)
    except Exception as e:
        print(f"Error running Flask app: {e}")
        sys.exit(1)

if __name__ == '__main__':
    print("Starting Lie Detection Data Viewer (Local Data Only)")
    print("=" * 50)
    print("Web app will be available at: http://localhost:8080")
    print("Press Ctrl+C to stop")
    print("")
    
    run_flask_app() 