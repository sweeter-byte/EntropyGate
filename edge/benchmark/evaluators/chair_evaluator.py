"""CHAIR evaluator — copied from the original EntropyGate codebase.

This is a direct copy of benchmark/evaluators/chair_evaluator.py.
See that file for full attribution and documentation.
"""

# Re-export from the original location to avoid code duplication
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from benchmark.evaluators.chair_evaluator import *
