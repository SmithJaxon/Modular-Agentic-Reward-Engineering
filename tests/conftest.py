"""
Summary: Shared pytest configuration for RewardLab tests.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
