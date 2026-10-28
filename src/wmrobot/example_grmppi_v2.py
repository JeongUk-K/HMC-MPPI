"""GR-MPPI unicycle navigation on a BARN occupancy map."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from solver.gr_mppi_v2 import GRMPPISolver
from src.wmrobot.common import (
    COLLISION_PENALTY,
    HORIZON,
    NUM_SAMPLES,
    TEMPERATURE,
    run_evaluation,
)


METHOD = "gr_mppi_v2"
DISPLAY_NAME = "GR-MPPI adaptive K"
NOISE_SIGMAS = ((0.125, 0.125), (0.25, 0.25), (0.5, 0.5))
MIN_SAMPLES_PER_COMPONENT = max(1, NUM_SAMPLES // 10)


def make_solver(model, collision_checker):
    if NUM_SAMPLES < 3 * MIN_SAMPLES_PER_COMPONENT:
        raise ValueError("WMROBOT_NUM_SAMPLES is too small for 3 components")
    return GRMPPISolver(
        model=model,
        horizon=HORIZON,
        num_samples=NUM_SAMPLES,
        temperature=TEMPERATURE,
        noise_sigmas=NOISE_SIGMAS,
        target_weights=(1.0, 1.0, 1.0),
        defensive_ratio=0.2,
        min_samples_per_component=MIN_SAMPLES_PER_COMPONENT,
        collision_checker=collision_checker,
        collision_penalty=COLLISION_PENALTY,
    )


if __name__ == "__main__":
    run_evaluation(
        METHOD,
        DISPLAY_NAME,
        make_solver,
        tuple(f"sigma={sigma[0]:g}" for sigma in NOISE_SIGMAS),
        {
            "noise_sigmas": [list(sigma) for sigma in NOISE_SIGMAS],
            "defensive_ratio": 0.2,
            "min_samples_per_component": MIN_SAMPLES_PER_COMPONENT,
        },
        adaptive=True,
    )
