#!/usr/bin/env python3

import os
import sys


ROOT = os.path.dirname(__file__)
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from agent_multi_cr.cli import main


if __name__ == "__main__":
    main()
