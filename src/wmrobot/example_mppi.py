"""Standard MPPI unicycle navigation on a BARN occupancy map."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from solver.mppi import MPPISolver
from src.wmrobot.common import (
    COLLISION_PENALTY,
    HORIZON,
    NUM_SAMPLES,
    TEMPERATURE,
    run_evaluation,
)


METHOD = "mppi"
DISPLAY_NAME = "MPPI sigma=0.25"
NOISE_SIGMA = (0.25, 0.25)


def make_solver(model, collision_checker):
    return MPPISolver(
        model=model,
        horizon=HORIZON,
        num_samples=NUM_SAMPLES,
        temperature=TEMPERATURE,
        noise_sigma=NOISE_SIGMA,
        collision_checker=collision_checker,
        collision_penalty=COLLISION_PENALTY,
    )


if __name__ == "__main__":
    run_evaluation(
        METHOD,
        DISPLAY_NAME,
        make_solver,
        ("sigma=0.25",),
        {"noise_sigma": list(NOISE_SIGMA)},
    )
