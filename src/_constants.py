"""
This file holds the project-wide constants in one place

We keep it tiny on purpose and with no other imports, so every other module
in the src/ folder (and also the top level scripts and notebooks) can import
RANDOM_STATE without making a circular import.
For example src/train_utils.py imports src/preprocessing.py, that imports
src/data_foundation.py, so if phase0 would import directly from train_utils
we will get a circular import. Putting the constant here just breaks that cycle.

Audit: this is the ONLY place in the project where RANDOM_STATE is bound to
the literal 42. Every other reference (train_*.py, phase*.py, the notebooks
via src/__init__.py) imports it from here.
"""

from __future__ import annotations

RANDOM_STATE: int = 42
