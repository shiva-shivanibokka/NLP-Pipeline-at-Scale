"""Pytest bootstrap.

Imports `configs.quiet` before anything pulls in `transformers`. Without this,
test modules that import the model or NER pipeline inherit whatever backend
`transformers` probes for: on a machine with TensorFlow installed it imports TF,
which on this project's Windows/conda setup fails with a numpy ABI error
(`numpy.dtype size changed`) that has nothing to do with the test. Every
`scripts/` entrypoint already does this; the test suite did not, so running the
suite and running the scripts behaved differently.

It also puts the repo root on `sys.path`, so `pytest tests/` works from the repo
root without an editable install.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import configs.quiet  # noqa: F401,E402  — side-effectful, must precede transformers
