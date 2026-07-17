# -*- coding: utf-8 -*-
"""Test-Paket. Fügt beim Import die Modulverzeichnisse zum sys.path hinzu."""

import os
import sys

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_TESTS_DIR, ".."))

for _sub in ("src/training", "src/runtime"):
    _p = os.path.join(_PROJECT_ROOT, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
