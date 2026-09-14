"""Force headless display config before any Panda display code is touched."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import config  # noqa: E402

config.bootstrap_display(headless=True)
