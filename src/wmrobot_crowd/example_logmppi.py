"""Run Log-MPPI on the shared moving-crowd scenario."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.wmrobot_crowd.example_grmppi_v2 import main


if __name__ == "__main__":
    main("logmppi")
