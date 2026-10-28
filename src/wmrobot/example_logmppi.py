"""Log-MPPI unicycle navigation on a BARN occupancy map."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np

from solver.logmppi import LogMPPISolver
from src.wmrobot.common import (
    COLLISION_PENALTY,
    HORIZON,
    NUM_SAMPLES,
    TEMPERATURE,
    run_evaluation,
)


METHOD = "logmppi"
DISPLAY_NAME = "Log-MPPI RMS sigma=0.25"
REFERENCE_SIGMA = 0.25
LOG_NOISE_MEAN = 1.023
LOG_NOISE_STD = 0.048**0.5
NOISE_SIGMA = REFERENCE_SIGMA / np.exp(LOG_NOISE_MEAN + LOG_NOISE_STD**2)


def make_solver(model, collision_checker):
    return LogMPPISolver(
        model=model,
        horizon=HORIZON,
        num_samples=NUM_SAMPLES,
        temperature=TEMPERATURE,
        noise_sigma=(NOISE_SIGMA, NOISE_SIGMA),
        log_noise_mean=LOG_NOISE_MEAN,
        log_noise_std=LOG_NOISE_STD,
        collision_checker=collision_checker,
        collision_penalty=COLLISION_PENALTY,
    )


if __name__ == "__main__":
    run_evaluation(
        METHOD,
        DISPLAY_NAME,
        make_solver,
        ("NLN RMS sigma=0.25",),
        {
            "reference_noise_rms": REFERENCE_SIGMA,
            "base_noise_sigma": float(NOISE_SIGMA),
            "log_noise_mean": LOG_NOISE_MEAN,
            "log_noise_std": LOG_NOISE_STD,
        },
    )
