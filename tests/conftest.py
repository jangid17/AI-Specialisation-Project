"""Point the app at a throwaway data directory before any of its modules are imported."""

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["RA_DATA_DIR"] = tempfile.mkdtemp(prefix="ra-tests-")
