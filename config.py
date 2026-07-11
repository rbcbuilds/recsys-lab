"""Convenience re-export so `import config` works from the project root.

The real settings live in ``recsys.config``; this thin shim lets scripts and
notebooks at the repo root grab paths without fussing with sys.path.
"""

from src.recsys.config import (  # noqa: F401
    DATA_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    RAW_DIR,
    Settings,
    settings,
)
