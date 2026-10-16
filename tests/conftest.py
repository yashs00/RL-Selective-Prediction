"""Pytest collection-time setup.

See scripts/run_all.py's identical header comment for the full explanation:
importing both lightgbm and torch in one process makes macOS's two bundled
libomp runtimes deadlock silently (0% CPU, no error, hangs forever) instead
of raising. conftest.py is imported by pytest before it collects any test
module, so this is the one place guaranteed to run before any test file
imports numpy/sklearn/lightgbm/torch.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
