"""Shared pytest fixtures/config.

Loads environment variables so that importing ``src.graph`` (which constructs
the ChatGroq client and Tavily tool at module load) succeeds. None of the tests
in this suite make network/API calls — they exercise pure routing, fusion, and
scoring logic only.
"""

import os
import sys

# Make the project root importable (so `import src.graph` works from anywhere).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv()
