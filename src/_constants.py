"""
The single source of truth for project-wide reproducibility constants.

This module is intentionally tiny and has no other imports. That property
matters: it lets every other module in `src/` (and the top-level CLI
scripts and notebooks) import `RANDOM_STATE` without creating a cycle.
Specifically, `src/train_utils.py` imports `src/phase1_preprocessing.py`,
which imports `src/phase0_data_foundation.py`, so if phase0 wanted to
import directly from train_utils we'd have a circular import. Putting
the constant here breaks that cycle cleanly.

Audit: this is the ONLY location in the project where `RANDOM_STATE` is
bound to the literal `42`. Every other reference (`train_*.py`,
`phase*.py`, the notebooks via `src/__init__.py`) imports it.
"""

from __future__ import annotations

RANDOM_STATE: int = 42
