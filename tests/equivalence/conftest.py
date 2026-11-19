import sys

import matplotlib
from _registry import REFERENCE_DIR

matplotlib.use("Agg")

if REFERENCE_DIR.exists():
    sys.path.insert(0, str(REFERENCE_DIR))
