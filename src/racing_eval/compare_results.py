import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = ("log_mppi", "gr_mppi_v2")
COLORS = ("tab:orange", "tab:blue")
SUMMARY_FIELDS = (
    "method",
    "collision_steps",
    "minimum_obstacle_clearance_m",
    "response_time_after_event_s",
    "recovery_time_after_pass_s",
    "settling_contour_std_m",
    "settling_steer_std_rad",
    "settling_accel_std_mps2",
    "settling_steer_jerk_rms",
    "action_saturation_rate",
    "ess_ratio_mean",
    "solver_time_mean_ms",
)


def load_results(result_root):
    results = []
    for method in METHODS:
        method_dir = result_root / method
        with (method_dir / "metrics.json").open(encoding="utf-8") as file:
            metrics = json.load(file)
        results.append((metrics, np.load(method_dir / "trajectory.npz")))
    return results


def add_phase_lines(axis, configuration):
    axis.axvline(
        configuration["event_time_s"], color="tab:red", linestyle="--", linewidth=1
    )
    axis.axvline(
        configuration["detour_end_time_s"],
        color="tab:blue",
        linestyle="--",
        linewidth=1,
    )
    axis.axvline(
        configuration["settling_start_time_s"],
        color="black",
        linestyle=":",
        linewidth=1,
    )


def save_comparison(results, output_path):
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    for (metrics, trajectory), color in zip(results, COLORS):
        label = metrics["display_name"]
        axes[0, 0].plot(
            trajectory["states"][:, 0], trajectory["states"][:, 1],
            color=color, label=label,
        )
        axes[0, 1].plot(
            trajectory["time_s"], np.abs(trajectory["states"][:, 1]),
            color=color, label=label,
        )
        axes[1, 0].plot(
            trajectory["action_time_s"], trajectory["actions"][:, 1],
            color=color, label=label,
        )
        axes[1, 1].plot(
            trajectory["action_time_s"],
            trajectory["ess"] / metrics["configuration"]["num_samples"],
            color=color, label=label,
        )
        axes[2, 0].plot(
            trajectory["action_time_s"], trajectory["sampling_rms"][:, 0],
            color=color, label=f"{label}: accel",
        )
        axes[2, 0].plot(
            trajectory["action_time_s"], trajectory["sampling_rms"][:, 1],
            color=color, linestyle="--", label=f"{label}: steer",
        )

    first_metrics, first_trajectory = results[0]
    obstacle = plt.Circle(
        first_trajectory["obstacle_center"],
        float(first_trajectory["obstacle_radius"]),
        color="tab:red",
        alpha=0.7,
    )
    axes[0, 0].add_patch(obstacle)
    axes[0, 0].axhline(0.0, color="gray", linestyle=":", label="original nominal")
    configuration = first_metrics["configuration"]
    axes[0, 0].plot(
        [
            configuration["reference_speed_mps"] * configuration["event_time_s"],
            configuration["reference_speed_mps"] * configuration["detour_end_time_s"],
        ],
        [configuration["detour_offset_m"]] * 2,
        "--",
        color="black",
        label="emergency nominal",
    )
    lane_width = 6.5
    axes[0, 0].set_ylim(-lane_width / 2 - 0.8, lane_width / 2 + 0.8)
    axes[0, 0].set_ylabel("y [m] (vertical scale enlarged)")
    axes[0, 0].set_title("Trajectory")
    axes[0, 1].set_title("Absolute lateral position")
    axes[1, 0].set_title("Steering input")
    axes[1, 1].set_title("ESS / K")
    axes[2, 0].set_title("Effective sampling RMS")

    gr_metrics, gr_trajectory = results[1]
    for component in range(gr_trajectory["sample_counts"].shape[1]):
        axes[2, 0].plot(
            gr_trajectory["action_time_s"],
            gr_trajectory["sample_counts"][:, component]
            / gr_metrics["configuration"]["num_samples"],
            linestyle=":",
            alpha=0.65,
            label=f"GR K{component + 1}/K",
        )
    axes[2, 0].set_title("Sampling RMS and GR allocation")

    rows = [
        ("Collision steps", "collision_steps", ".0f"),
        ("Minimum clearance [m]", "minimum_obstacle_clearance_m", ".3f"),
        ("Recovery after pass [s]", "recovery_time_after_pass_s", ".3f"),
        ("Settling |contour| [m]", "settling_contour_mean_abs_m", ".4f"),
        ("Settling steer std [rad]", "settling_steer_std_rad", ".5f"),
        ("Settling steer jerk RMS", "settling_steer_jerk_rms", ".5f"),
    ]
    table_text = []
    for title, field, spec in rows:
        values = []
        for metrics, _ in results:
            value = metrics[field]
            values.append("-" if value is None else format(value, spec))
        table_text.append([title, *values])
    axes[2, 1].axis("off")
    axes[2, 1].table(
        cellText=table_text,
        colLabels=["Metric", *(m["display_name"] for m, _ in results)],
        cellLoc="center",
        loc="center",
    ).scale(1.0, 1.6)
    axes[2, 1].set_title("Transient and settling metrics")

    for axis in (axes[0, 1], axes[1, 0], axes[1, 1], axes[2, 0]):
        add_phase_lines(axis, first_metrics["configuration"])
    for axis in axes.flat:
        axis.grid(True, alpha=0.3)
    for axis in axes.flat[:-1]:
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def write_summary(results, result_root):
    with (result_root / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for metrics, _ in results:
            writer.writerow({field: metrics.get(field) for field in SUMMARY_FIELDS})

    lines = [
        "# Racing sudden-obstacle evaluation",
        "",
        "| Method | Collision steps | Min clearance [m] | Response [s] | Recovery [s] | Settling contour std [m] | Settling steer std [rad] | Steer jerk RMS | ESS/K |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metrics, _ in results:
        recovery = metrics["recovery_time_after_pass_s"]
        response = metrics["response_time_after_event_s"]
        lines.append(
            f'| {metrics["display_name"]} | {metrics["collision_steps"]} '
            f'| {metrics["minimum_obstacle_clearance_m"]:.3f} '
            f'| {"-" if response is None else f"{response:.2f}"} '
            f'| {"-" if recovery is None else f"{recovery:.2f}"} '
            f'| {metrics["settling_contour_std_m"]:.4f} '
            f'| {metrics["settling_steer_std_rad"]:.5f} '
            f'| {metrics["settling_steer_jerk_rms"]:.5f} '
            f'| {metrics["ess_ratio_mean"]:.4f} |'
        )
    gr_metrics = next(metrics for metrics, _ in results if metrics["method"] == "gr_mppi_v2")
    allocation = gr_metrics["sample_allocation_ratio"]
    lines += [
        "",
        "GR-MPPI K1/K2/K3 allocation ratio:",
        f'- pre-event mean: {[round(value, 3) for value in allocation["pre_event_mean"]]}',
        f'- lane-change peak: {[round(value, 3) for value in allocation["event_peak"]]}',
        f'- nominal-return peak: {[round(value, 3) for value in allocation["return_peak"]]}',
        f'- settling mean: {[round(value, 3) for value in allocation["settling_mean"]]}',
        "",
        "Red dashed line: obstacle/lane-change event. Blue dashed line: nominal returns. Black dotted line: settling window start.",
        "At obstacle appearance, the local nominal changes from the center line to the same adjacent lane for both methods.",
        "Both methods use the same model, cost, horizon, sample budget, temperature, seed, warm-start, and initial per-action noise RMS.",
        "Use multiple seeds before drawing statistical conclusions.",
    ]
    (result_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    result_root = Path(__file__).resolve().parents[2] / "result/racing_eval"
    results = load_results(result_root)
    save_comparison(results, result_root / "comparison.png")
    write_summary(results, result_root)
    print((result_root / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
