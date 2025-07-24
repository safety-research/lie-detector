# Re-export the Flask app for Vercel
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the actual app
from app import app as flask_app

# Export for Vercel
app = flask_app