"""
Pytest configuration file.
"""

import os
import sys

# Add the project root to the Python path so tests can import modules properly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Additional pytest configuration can be added here 