"""
tests/conftest.py - Pytest Configuration & System Path Failsafe
==============================================================
Ensures project root is always present in sys.path regardless of
whether pytest is invoked via 'pytest' or 'python -m pytest' on Linux/Windows.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
