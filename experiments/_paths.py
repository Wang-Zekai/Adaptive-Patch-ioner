"""Paths shared by the experiment entry points."""

from pathlib import Path
import sys


EXPERIMENTS = Path(__file__).resolve().parent
ROOT = EXPERIMENTS.parent
CONFIGS = ROOT / "configs"
sys.path.insert(1, str(ROOT / "src"))
