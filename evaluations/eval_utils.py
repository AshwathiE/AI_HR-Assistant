"""
evaluations/utils.py
--------------------
Utility functions and wrappers for the RAG Evaluation Pipeline.
Reuses existing preprocessing, embedding generation, vector retrieval, LLM generation,
and caching logic from the server module without altering any existing server code.
"""

import os
import sys
import time
from typing import List, Dict, Any, Tuple, Optional

# Set up project paths
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(EVAL_DIR)
SERVER_DIR = os.path.join(PROJECT_ROOT, "server")
REPORTS_DIR = os.path.join(EVAL_DIR, "reports")
DATASET_PATH = os.path.join(EVAL_DIR, "evaluation_dataset.json")

if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(1, PROJECT_ROOT)

# Import existing backend modules from server

class Timer:
    """Context manager for high-precision execution timing."""
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start
