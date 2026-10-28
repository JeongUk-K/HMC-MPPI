from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = (
    ("mppi", "MPPI"),
    ("logmppi", "Log-MPPI"),
    ("gr_mppi_v2", "GR-MPPI"),
)


def main() -> None:
    result_root = Path(__file__).resolve().parents[2] / "result/furuta_pendulum"
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for method, label in METHODS:
        path = result_root / method / "diagnostics.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run furuta_epndulum.sh first")
        with np.load(path) as data:
            axes[0].plot(data["time_s"], data["ess_ratio"], label=label)
            axes[1].plot(data["time_s"], data["predicted_cost"], label=label)

    axes[0].set_ylabel("ESS / K")
    axes[0].set_yscale("log")
    axes[0].set_ylim(1.0e-4, 1.0)
    axes[1].set_yscale("symlog", linthresh=1.0)
    axes[1].set_ylim(bottom=0.0)
    axes[1].set_ylabel("Minimum cost")
    axes[1].set_xlabel("Time [s]")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.suptitle("Furuta pendulum solver comparison")
    fig.tight_layout()
    output_path = result_root / "diagnostics_comparison.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"[saved] {output_path}")


if __name__ == "__main__":
    main()
