#!/usr/bin/env python3
"""Thin entry script to trigger the pipeline module."""

import sys
from pathlib import Path

# Provide resilience if someone runs this from outside the project dir
scripts_dir = Path(__file__).resolve().parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from pipeline import main

if __name__ == "__main__":
    sys.exit(main())
