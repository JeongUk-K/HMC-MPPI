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
    "method", "display_name", "position_rmse_m", "position_mean_error_m",
    "position_max_error_m", "velocity_rmse_mps", "yaw_rmse_rad",
    "control_total_variation", "collision_steps", "solver_time_mean_ms",
    "solver_time_p95_ms", "ess_ratio_mean", "sample_count_mean",
)


def load_results(result_root):
    results = []
    for method in METHODS:
        with (result_root / method / "metrics.json").open(encoding="utf-8") as file:
            metrics = json.load(file)
        results.append((metrics, np.load(result_root / method / "trajectory.npz")))
    return results


def save_comparison(results, output_path):
    colors = ("tab:green", "tab:blue", "tab:orange")
    labels = [metrics["display_name"] for metrics, _ in results]
    figure = plt.figure(figsize=(17, 10))
    trajectory_axis = figure.add_subplot(2, 3, 1, projection="3d")
    error_axis = figure.add_subplot(2, 3, 2)
    ess_axis = figure.add_subplot(2, 3, 3)
    for (metrics, trajectory), color in zip(results, colors):
        states = trajectory["states"]
        trajectory_axis.plot(*states[:, :3].T, color=color, label=metrics["display_name"])
        error_axis.plot(
            trajectory["time_s"], trajectory["position_error"],
            color=color, label=metrics["display_name"],
        )
        ess_axis.plot(
            trajectory["time_s"][:-1], trajectory["ess"] / metrics["configuration"]["num_samples"],
            color=color, label=metrics["display_name"],
        )
    reference = results[0][1]["references"]
    trajectory_axis.plot(*reference[:, :3].T, "--", color="gray", label="reference")
    trajectory_axis.set_title("tracking_zigzag trajectory")
    error_axis.set_title("Position error [m]")
    ess_axis.set_title("ESS / K")
    bar_fields = (
        ("position_rmse_m", "Position RMSE [m]"),
        ("control_total_variation", "Control total variation"),
        ("solver_time_mean_ms", "Solver time [ms]"),
    )
    x = np.arange(len(results))
    for plot_index, (field, title) in enumerate(bar_fields, start=4):
        axis = figure.add_subplot(2, 3, plot_index)
        values = [metrics[field] for metrics, _ in results]
        bars = axis.bar(x, values, color=colors)
        axis.set_xticks(x, labels, rotation=15)
        axis.set_title(title)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3g}", ha="center", va="bottom")
        axis.grid(True, alpha=0.3)
    for axis in (trajectory_axis, error_axis, ess_axis):
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def quaternion_rotation(quaternion):
    x, y, z, w = quaternion / np.linalg.norm(quaternion)
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ])


def save_video(metrics, trajectory, output_path):
    states = trajectory["states"]
    actions = trajectory["actions"]
    predictions = trajectory["predicted_states"]
    reference = trajectory["references"]
    limits = np.vstack((states[:, :3], reference[:, :3]))
    center = 0.5 * (limits.min(axis=0) + limits.max(axis=0))
    radius = max(1.5, 0.6 * np.ptp(limits, axis=0).max())
    figure = plt.figure(figsize=(9, 8))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(*reference[:, :3].T, "--", color="gray", linewidth=1, label="reference")
    trace, = axis.plot([], [], [], color="tab:blue", linewidth=2, label="actual")
    prediction, = axis.plot([], [], [], "--", color="deepskyblue", linewidth=1.5, label="predicted X*")
    target, = axis.plot([], [], [], "o", color="tab:green", markersize=6, label="target")
    body_axes = [axis.plot([], [], [], color=color, linewidth=3)[0] for color in ("red", "green", "blue")]
    axis.set(
        xlim=(center[0] - radius, center[0] + radius),
        ylim=(center[1] - radius, center[1] + radius),
        zlim=(center[2] - radius, center[2] + radius),
        xlabel="x [m]", ylabel="y [m]", zlabel="z [m]",
    )
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=25, azim=45)
    axis.legend(loc="upper right")
    dt = metrics["configuration"]["dt_s"]
    writer = FFMpegWriter(fps=round(1.0 / dt), bitrate=2000)
    with writer.saving(figure, str(output_path), dpi=110):
        for index, state in enumerate(states):
            trace.set_data(states[:index + 1, 0], states[:index + 1, 1])
            trace.set_3d_properties(states[:index + 1, 2])
            if index < len(predictions):
                prediction.set_data(predictions[index, :, 0], predictions[index, :, 1])
                prediction.set_3d_properties(predictions[index, :, 2])
            else:
                prediction.set_data([], [])
                prediction.set_3d_properties([])
            target.set_data([reference[index, 0]], [reference[index, 1]])
            target.set_3d_properties([reference[index, 2]])
            rotation = quaternion_rotation(state[3:7])
            for body_axis, direction in zip(body_axes, rotation.T):
                endpoint = state[:3] + 0.18 * direction
                body_axis.set_data([state[0], endpoint[0]], [state[1], endpoint[1]])
                body_axis.set_3d_properties([state[2], endpoint[2]])
            action = actions[min(index, len(actions) - 1)]
            axis.set_title(
                f'{metrics["display_name"]}  t={index * dt:.2f}s  '
                f'u=[{action[0]:+.2f}, {action[1]:+.2f}, {action[2]:+.2f}, {action[3]:+.2f}]'
            )
            writer.grab_frame()
    plt.close(figure)


def write_summary(results, result_root):
    with (result_root / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for metrics, _ in results:
            writer.writerow({field: metrics.get(field) for field in FIELDS})
    lines = [
        "# Quadrotor tracking_zigzag evaluation", "",
        "| Method | Position RMSE/Max [m] | Velocity RMSE [m/s] | Yaw RMSE [rad] | Control TV | Collision steps | Solver mean/p95 [ms] | ESS/K | Mean K |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metrics, _ in results:
        lines.append(
            f'| {metrics["display_name"]} '
            f'| {metrics["position_rmse_m"]:.3f}/{metrics["position_max_error_m"]:.3f} '
            f'| {metrics["velocity_rmse_mps"]:.3f} '
            f'| {metrics["yaw_rmse_rad"]:.3f} '
            f'| {metrics["control_total_variation"]:.2f} '
            f'| {metrics["collision_steps"]} '
            f'| {metrics["solver_time_mean_ms"]:.3f}/{metrics["solver_time_p95_ms"]:.3f} '
            f'| {metrics["ess_ratio_mean"]:.4f} '
            f'| {metrics.get("sample_count_mean", "-")} |'
        )
    lines += [
        "",
        "세 방법은 동일한 zigzag reference, 관측 잡음, seed, K, horizon, temperature를 사용한다.",
        "측정 시간에는 JIT compile, ESS, 예측 rollout, plant dynamics, metric/plot/I/O가 포함되지 않는다.",
    ]
    if any(metrics["configuration"].get("is_smoke_test") for metrics, _ in results):
        lines.append("주의: 현재 저장된 결과는 축소 smoke test이며 전체 기본 평가 결과가 아니다.")
    (result_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    result_root = Path(__file__).resolve().parents[2] / "result/quadrotor_eval"
    results = load_results(result_root)
    save_comparison(results, result_root / "comparison.png")
    write_summary(results, result_root)
    for metrics, trajectory in results:
        output = result_root / metrics["method"] / "quadrotor.mp4"
        save_video(metrics, trajectory, output)
        print(f"[saved] {output}")
    print((result_root / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
