"""Compare MPPI variants on the shared moving-crowd scenario."""

import csv
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np

from src.wmrobot_crowd.example_grmppi_v2 import (
    GOAL_STATE,
    NUM_SAMPLES,
    PEDESTRIAN_PARAMS,
    SAFE_DISTANCE,
    pedestrian_positions_np,
)


METHODS = ("mppi", "logmppi", "gr_mppi_v2")
FIELDS = (
    "method",
    "display_name",
    "success",
    "collision",
    "steps",
    "travel_time_s",
    "path_length_m",
    "final_position_error_m",
    "minimum_pedestrian_clearance_m",
    "solver_time_mean_ms",
    "solver_time_p95_ms",
    "ess_mean",
    "ess_ratio_mean",
)


def load_results(result_root):
    results = []
    for method in METHODS:
        method_dir = result_root / method
        metrics_path = method_dir / "metrics.json"
        trajectory_path = method_dir / "trajectory.npz"
        if not metrics_path.is_file() or not trajectory_path.is_file():
            raise FileNotFoundError(
                f"Missing result for {method}; run wmrobot_crowd.sh first"
            )
        with metrics_path.open(encoding="utf-8") as file:
            metrics = json.load(file)
        results.append((metrics, np.load(trajectory_path)))
    return results


def save_plot(results, output_path):
    colors = ("tab:blue", "tab:orange", "tab:green")
    fig, axes = plt.subplots(3, 2, figsize=(15, 13))
    max_time = max(item[1]["states"][-1, 3] for item in results)
    crowd_times = np.linspace(0.0, max_time, 300)
    crowd_paths = np.asarray([pedestrian_positions_np(t) for t in crowd_times])
    for pedestrian in range(len(PEDESTRIAN_PARAMS)):
        axes[0, 0].plot(
            crowd_paths[:, pedestrian, 0],
            crowd_paths[:, pedestrian, 1],
            color="gray",
            alpha=0.18,
        )

    for (metrics, trajectory), color in zip(results, colors):
        states = trajectory["states"]
        times = states[:, 3]
        label = metrics["display_name"]
        axes[0, 0].plot(states[:, 0], states[:, 1], label=label, color=color)
        axes[0, 1].plot(times, states[:, 0], label=label, color=color)
        axes[1, 0].plot(times, states[:, 1], label=label, color=color)
        axes[1, 1].plot(
            times,
            trajectory["pedestrian_clearance_m"],
            label=label,
            color=color,
        )
        axes[2, 0].plot(
            times[1:],
            trajectory["ess"] / NUM_SAMPLES,
            label=label,
            color=color,
        )

    axes[0, 0].axhline(0.0, color="black", linestyle="--", label="reference")
    axes[0, 0].scatter(*GOAL_STATE[:2], marker="*", s=140, color="red")
    axes[0, 0].set(xlim=(-0.4, 10.4), ylim=(-2.1, 2.1), xlabel="x [m]", ylabel="y [m]")
    axes[0, 0].set_aspect("equal")
    axes[0, 0].set_title("robot paths and pedestrian paths")
    axes[0, 0].legend(ncol=2, fontsize=8)

    axes[0, 1].set(title="longitudinal progress", ylabel="x [m]")
    axes[1, 0].axhline(0.0, color="black", linestyle="--")
    axes[1, 0].set(title="straight-path tracking", ylabel="lateral y [m]")
    axes[1, 1].axhline(
        SAFE_DISTANCE, color="black", linestyle=":", label="collision distance"
    )
    axes[1, 1].set(title="nearest pedestrian distance", ylabel="distance [m]")
    axes[2, 0].set(title="effective sample size", ylabel="ESS / K", ylim=(0, 1.05))

    labels = [item[0]["display_name"] for item in results]
    means = [item[0]["solver_time_mean_ms"] for item in results]
    p95 = [item[0]["solver_time_p95_ms"] for item in results]
    x_positions = np.arange(len(results))
    axes[2, 1].bar(x_positions, means, color=colors)
    axes[2, 1].scatter(x_positions, p95, color="black", marker="_", label="p95")
    axes[2, 1].set_xticks(x_positions, labels, rotation=15)
    axes[2, 1].set(title="solver time", ylabel="ms / control step")
    axes[2, 1].legend()

    for axis in axes.flat:
        axis.grid(True, alpha=0.3)
        if axis not in (axes[0, 0], axes[2, 1]):
            axis.set_xlabel("time [s]")
            axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_summaries(results, result_root):
    with (result_root / "comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for metrics, _ in results:
            writer.writerow({field: metrics[field] for field in FIELDS})

    prediction_horizon = results[0][0]["configuration"][
        "pedestrian_prediction_horizon_s"
    ]
    prediction_model = results[0][0]["configuration"][
        "pedestrian_prediction"
    ]
    lines = [
        "# Moving-crowd solver comparison",
        "",
        f"Pedestrian prediction horizon: `{prediction_horizon:g} s` "
        "(positions are frozen after this point).",
        f"Prediction model: {prediction_model}.",
        "",
        "| Method | Result | Time [s] | Path [m] | Min clearance [m] | Solver mean/p95 [ms] | ESS/K |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for metrics, _ in results:
        result = (
            "success"
            if metrics["success"]
            else "collision"
            if metrics["collision"]
            else "timeout"
        )
        lines.append(
            f'| {metrics["display_name"]} | {result} '
            f'| {metrics["travel_time_s"]:.2f} '
            f'| {metrics["path_length_m"]:.3f} '
            f'| {metrics["minimum_pedestrian_clearance_m"]:.3f} '
            f'| {metrics["solver_time_mean_ms"]:.3f} / '
            f'{metrics["solver_time_p95_ms"]:.3f} '
            f'| {metrics["ess_ratio_mean"]:.4f} |'
        )
    lines.extend(
        [
            "",
            "All methods use the same crowd, reference, seed, horizon, sample count, and temperature.",
            "MPPI and Log-MPPI have matched RMS noise `(0.25, 0.5)`; GR-MPPI uses its three-component mixture.",
            "This is one deterministic crowd realization and one controller seed.",
        ]
    )
    (result_root / "comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main():
    result_root = Path(__file__).resolve().parents[2] / "result" / "wmrobot_crowd"
    results = load_results(result_root)
    save_plot(results, result_root / "comparison.png")
    write_summaries(results, result_root)
    print(f"[saved] {result_root / 'comparison.png'}")
    print(f"[saved] {result_root / 'comparison.md'}")


if __name__ == "__main__":
    main()
