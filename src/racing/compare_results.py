import csv
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np


METHODS = ("mppi", "gr_mppi_v2", "log_mppi")
FIELDS = (
    "method", "display_name", "goal_reached", "goal_time_s",
    "path_progress_percent", "collision_steps", "collision_rate",
    "contour_rmse_m", "lag_rmse_m", "speed_rmse_mps",
    "accel_rms_mps2", "steer_rms_rad", "control_total_variation",
    "solver_time_mean_ms", "solver_time_p95_ms", "ess_ratio_mean",
    "sample_count_mean",
)


def load_results(result_root):
    results = []
    for method in METHODS:
        with (result_root / method / "metrics.json").open(encoding="utf-8") as file:
            metrics = json.load(file)
        results.append((metrics, np.load(result_root / method / "trajectory.npz")))
    return results


def save_comparison(results, output_path):
    labels = [metrics["display_name"] for metrics, _ in results]
    colors = ("tab:green", "tab:blue", "tab:orange")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for (metrics, trajectory), color in zip(results, colors):
        axes[0, 0].plot(
            trajectory["states"][:, 0], trajectory["states"][:, 1],
            label=metrics["display_name"], color=color,
        )
        axes[0, 1].plot(
            trajectory["time_s"], np.abs(trajectory["contour_error"]),
            label=metrics["display_name"], color=color,
        )
        axes[0, 2].plot(
            trajectory["time_s"][1:], trajectory["ess"] / metrics["configuration"]["num_samples"],
            label=metrics["display_name"], color=color,
        )
    path = results[0][1]["path"]
    axes[0, 0].plot(path[:, 0], path[:, 1], "--", color="gray", label="reference")
    for x, y, radius in results[0][1]["dynamic_circles"]:
        axes[0, 0].add_patch(
            plt.Circle((x, y), radius, color="tab:red", alpha=0.7)
        )
    for appearance_step in results[0][1]["dynamic_appearance_steps"]:
        event_time = appearance_step * results[0][0]["configuration"]["dt_s"]
        axes[0, 1].axvline(event_time, color="black", linestyle="--", linewidth=0.8)
        axes[0, 2].axvline(event_time, color="black", linestyle="--", linewidth=0.8)
    axes[0, 0].set_aspect("equal")
    axes[0, 0].set_title("Trajectory")
    axes[0, 1].set_title("Absolute contour error")
    axes[0, 2].set_title("ESS / K")

    x = np.arange(len(labels))
    fields = (
        ("goal_time_s", "Goal time [s]"),
        ("lag_rmse_m", "Lag RMSE [m]"),
        ("control_total_variation", "Control total variation"),
    )
    for axis, (field, title) in zip(axes[1], fields):
        values = [metrics[field] for metrics, _ in results]
        bars = axis.bar(x, values, color=colors)
        axis.set_xticks(x, labels, rotation=15)
        axis.set_title(title)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3g}", ha="center", va="bottom")
    for axis in axes.flat:
        axis.grid(True, alpha=0.3)
    for axis in axes[0]:
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_video(metrics, trajectory, output_path):
    states = trajectory["states"]
    actions = trajectory["actions"]
    predicted_states = trajectory["predicted_states"]
    path = trajectory["path"]
    circles = trajectory["circles"]
    static_circle_count = int(trajectory["static_circle_count"])
    dynamic_circles = trajectory["dynamic_circles"]
    appearance_steps = trajectory["dynamic_appearance_steps"]
    lane_width = 6.5
    normal = np.column_stack(
        (np.cos(path[:, 2] + np.pi / 2), np.sin(path[:, 2] + np.pi / 2))
    )
    fig, axis = plt.subplots(figsize=(8, 8))
    axis.plot(path[:, 0], path[:, 1], "--", color="gray", linewidth=1)
    axis.plot(*(path[:, :2] + lane_width / 2 * normal).T, color="green", linewidth=0.8)
    axis.plot(*(path[:, :2] - lane_width / 2 * normal).T, color="green", linewidth=0.8)
    for x, y, radius in circles[:static_circle_count]:
        axis.add_patch(plt.Circle((x, y), radius, color="gray"))
    dynamic_patches = []
    for x, y, radius in dynamic_circles:
        patch = plt.Circle((x, y), radius, color="tab:red", zorder=20)
        patch.set_visible(False)
        axis.add_patch(patch)
        dynamic_patches.append(patch)
    trace, = axis.plot([], [], color="tab:blue", linewidth=1.5, label="driven")
    prediction, = axis.plot(
        [], [], color="deepskyblue", linestyle="--", linewidth=2,
        label="predicted X*", zorder=15,
    )
    robot, = axis.plot([], [], "o", color="tab:red", markersize=7)
    heading, = axis.plot([], [], color="tab:red", linewidth=2)
    axis.set(xlim=(-40, 40), ylim=(-40, 40), xlabel="x [m]", ylabel="y [m]")
    axis.set_aspect("equal")
    axis.grid(True, alpha=0.2)
    axis.legend(loc="upper right")
    writer = FFMpegWriter(fps=10, bitrate=1800)
    with writer.saving(fig, str(output_path), dpi=110):
        for index, state in enumerate(states):
            for patch, appearance_step in zip(dynamic_patches, appearance_steps):
                patch.set_visible(index >= appearance_step)
            trace.set_data(states[: index + 1, 0], states[: index + 1, 1])
            if index < len(predicted_states):
                prediction.set_data(
                    predicted_states[index, :, 0],
                    predicted_states[index, :, 1],
                )
            else:
                prediction.set_data([], [])
            robot.set_data([state[0]], [state[1]])
            heading.set_data(
                [state[0], state[0] + 1.5 * np.cos(state[2])],
                [state[1], state[1] + 1.5 * np.sin(state[2])],
            )
            action = actions[min(index, len(actions) - 1)]
            axis.set_title(
                f'{metrics["display_name"]}  t={index * 0.1:.1f}s  '
                f'v={state[3]:.2f}  a={action[0]:+.2f}  steer={action[1]:+.3f}'
            )
            writer.grab_frame()
    plt.close(fig)


def write_summary(results, result_root):
    with (result_root / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for metrics, _ in results:
            writer.writerow({field: metrics.get(field) for field in FIELDS})
    lines = [
        "# Racing evaluation summary", "",
        "| Method | Goal time [s] | Collision rate | Contour/Lag RMSE [m] | Speed RMSE [m/s] | Control TV | Solver mean/p95 [ms] | ESS/K | Mean K |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metrics, _ in results:
        lines.append(
            f'| {metrics["display_name"]} | {metrics["goal_time_s"]:.2f} '
            f'| {metrics["collision_rate"]:.4f} '
            f'| {metrics["contour_rmse_m"]:.3f}/{metrics["lag_rmse_m"]:.3f} '
            f'| {metrics["speed_rmse_mps"]:.3f} '
            f'| {metrics["control_total_variation"]:.1f} '
            f'| {metrics["solver_time_mean_ms"]:.3f}/{metrics["solver_time_p95_ms"]:.3f} '
            f'| {metrics["ess_ratio_mean"]:.4f} '
            f'| {metrics.get("sample_count_mean", "-")} |'
        )
    lines += [
        "",
        "단일 seed 결과이며 세 방법 모두 같은 track, obstacle map, K, horizon을 사용한다.",
        "Reference path 전방 10m, 8m, 6m에 반지름 1.1m 장애물이 순차적으로 등장한다.",
    ]
    (result_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    result_root = Path(__file__).resolve().parents[2] / "result/racing"
    results = load_results(result_root)
    save_comparison(results, result_root / "comparison.png")
    write_summary(results, result_root)
    for metrics, trajectory in results:
        output = result_root / metrics["method"] / "racing.mp4"
        save_video(metrics, trajectory, output)
        print(f"[saved] {output}")
    print((result_root / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
