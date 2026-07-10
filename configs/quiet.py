"""
Import for side effect at process entry, BEFORE `transformers` is imported.

`transformers` eagerly probes for TensorFlow/Flax and, on some setups, floods
stderr with `MessageFactory`/oneDNN chatter that buries real tracebacks. We only
use the PyTorch backend, so switch the others off up front.
"""

import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Load pyarrow/datasets FIRST, before torch. On this Windows/conda setup importing
# pyarrow.dataset *after* torch+sklearn+transformers are resident segfaults (native
# library-init conflict). Every experiment entrypoint imports this module before any
# torch import, so eagerly loading datasets here pins the safe ordering.
try:
    import datasets  # noqa: F401
except Exception:
    pass  # datasets not installed (e.g. API-only image) — nothing to pre-load
