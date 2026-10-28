from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def normalized_ess(weights: np.ndarray) -> float:
    weights = np.asarray(weights)
    return float(1.0 / (weights.size * np.sum(weights**2)))


def save_diagnostics(
    result_dir: Path,
    display_name: str,
    time_s: np.ndarray,
    ess_ratio: np.ndarray,
    predicted_cost: np.ndarray,
) -> None:
    np.savez_compressed(
        result_dir / "diagnostics.npz",
        time_s=time_s,
        ess_ratio=ess_ratio,
        predicted_cost=predicted_cost,
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(time_s, ess_ratio, color="tab:purple")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("ESS / K")
    axes[0].set_ylim(1.0e-4, 1.0)

    axes[1].plot(time_s, predicted_cost, color="tab:red")
    axes[1].set_yscale("symlog", linthresh=1.0)
    axes[1].set_ylim(bottom=0.0)
    axes[1].set_ylabel("Minimum cost")
    axes[1].set_xlabel("Time [s]")

    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.suptitle(f"{display_name} sampling diagnostics")
    fig.tight_layout()
    fig.savefig(result_dir / "diagnostics.png", dpi=150)
    plt.close(fig)
