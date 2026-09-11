#!/usr/bin/env python3
"""
Inspectra AI — Complete TensorFlow / Keras Pipeline Verification Suite
=====================================================================
Automated validation of Requirements 1-27 (training/test_keras.py).
"""

import sys
from pathlib import Path

# Delegate to root test_keras.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from test_keras import main

if __name__ == "__main__":
    sys.exit(main())
