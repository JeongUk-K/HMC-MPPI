import csv
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np

from src.cartpole.cartpole import CartPole


METHOD_ORDER = (
    "mppi_sigma5",
    "mppi_sigma10",
    "mppi_sigma15",
    "gr_mppi",
    "gr_mppi_v2",
    "log_mppi",
)
CSV_FIELDS = (
    "method",
    "display_name",
    "success",
    "initial_swingup_success",
    "initial_settling_time_s",
    "recovery_success",
    "recovery_time_s",
    "disturbance_start_time_s",
    "disturbance_duration_s",
    "disturbance_end_time_s",
    "disturbance_theta_ddot_rad_s2",
    "settling_time_s",
    "transient_weighted_iae",
    "recovery_weighted_iae",
    "disturbance_response_total_weighted_iae",
    "peak_disturbance_response_weighted_error",
    "total_weighted_ise",
    "steady_weighted_rmse",
    "steady_weighted_mean",
    "steady_weighted_p95",
    "final_2s_weighted_rmse",
    "max_abs_cart_position_m",
    "max_abs_pole_rate_rad_s",
    "control_rms_n",
    "control_energy_n2_s",
    "disturbance_response_control_energy_n2_s",
    "control_max_abs_n",
    "control_total_variation_n",
    "solver_time_mean_ms",
    "solver_time_p95_ms",
    "solver_time_max_ms",
    "solver_frequency_hz",
    "ess_mean",
    "ess_min",
    "ess_p05",
    "ess_ratio_mean",
)


def load_results(result_root):
    results = []
    for method in METHOD_ORDER:
        metrics_path = result_root / method / "metrics.json"
        trajectory_path = result_root / method / "trajectory.npz"
        if not metrics_path.exists() or not trajectory_path.exists():
            raise FileNotFoundError(
                f"Missing evaluation output for {method}: run cartpole_eval.sh"
            )
        with metrics_path.open(encoding="utf-8") as file:
            metrics = json.load(file)
        results.append((metrics, np.load(trajectory_path)))
    return results


def write_csv(results, output_path):
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for metrics, _ in results:
            writer.writerow({field: metrics.get(field) for field in CSV_FIELDS})


def annotate_bars(axis, bars, values, fmt="{:.2f}"):
    for bar, value in zip(bars, values):
        if np.isfinite(value):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                fmt.format(value),
                ha="center",
                va="bottom",
                fontsize=8,
            )


def save_comparison_plot(results, output_path):
    metrics = [item[0] for item in results]
    trajectories = [item[1] for item in results]
    labels = [item["display_name"] for item in metrics]
    x_positions = np.arange(len(labels))
    colors = plt.cm.tab10(np.arange(len(labels)))

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    for label, trajectory, color in zip(labels, trajectories, colors):
        axes[0, 0].plot(
            trajectory["time_s"],
            trajectory["weighted_error"],
            label=label,
            color=color,
        )
    axes[0, 0].set_title("Weighted state error trajectory")
    axes[0, 0].set_xlabel("time [s]")
    axes[0, 0].set_ylabel(r"$\sqrt{e^T Q e}$")
    axes[0, 0].axvline(
        metrics[0]["disturbance_start_time_s"],
        color="black",
        linestyle="-",
        label="disturbance start",
    )
    axes[0, 0].axvline(
        metrics[0]["disturbance_end_time_s"],
        color="black",
        linestyle="--",
        label="disturbance end",
    )
    axes[0, 0].grid(True)
    axes[0, 0].legend(fontsize=8)

    max_recovery_time = (
        metrics[0]["configuration"]["duration_s"]
        - metrics[0]["disturbance_end_time_s"]
    )
    recovery_times = np.array(
        [
            item["recovery_time_s"]
            if item["recovery_time_s"] is not None
            else max_recovery_time
            for item in metrics
        ],
        dtype=float,
    )
    bars = axes[0, 1].bar(x_positions, recovery_times, color=colors)
    for bar, item in zip(bars, metrics):
        if not item["recovery_success"]:
            bar.set_hatch("//")
            axes[0, 1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                "FAIL",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axes[0, 1].set_title("Recovery after sustained disturbance (hatched = failed)")
    axes[0, 1].set_ylabel("time [s]")

    recovery_iae = np.array(
        [item["recovery_weighted_iae"] for item in metrics]
    )
    bars = axes[1, 0].bar(x_positions, recovery_iae, color=colors)
    annotate_bars(axes[1, 0], bars, recovery_iae)
    axes[1, 0].set_title("Weighted IAE until recovery")
    axes[1, 0].set_ylabel("weighted error·s")

    control_energy = np.array(
        [item["disturbance_response_control_energy_n2_s"] for item in metrics]
    )
    bars = axes[1, 1].bar(x_positions, control_energy, color=colors)
    annotate_bars(axes[1, 1], bars, control_energy, "{:.0f}")
    axes[1, 1].set_title("Disturbance-response control energy")
    axes[1, 1].set_ylabel(r"$\sum u^2 \Delta t$ [N$^2$s]")

    solver_time = np.array(
        [item["solver_time_mean_ms"] for item in metrics]
    )
    solver_p95 = np.array(
        [item["solver_time_p95_ms"] for item in metrics]
    )
    bars = axes[2, 0].bar(x_positions, solver_time, color=colors)
    axes[2, 0].scatter(
        x_positions,
        solver_p95,
        marker="_",
        color="black",
        label="p95",
    )
    annotate_bars(axes[2, 0], bars, solver_time)
    axes[2, 0].set_title("Pure solver time (mean bars, p95 marks)")
    axes[2, 0].set_ylabel("time [ms]")
    axes[2, 0].legend()

    ess_ratio = np.array([item["ess_ratio_mean"] for item in metrics])
    bars = axes[2, 1].bar(x_positions, ess_ratio, color=colors)
    annotate_bars(axes[2, 1], bars, ess_ratio, "{:.3f}")
    axes[2, 1].set_title("Mean ESS / K")
    axes[2, 1].set_ylabel("ratio")
    axes[2, 1].set_ylim(0.0, max(1.0, ess_ratio.max() * 1.15))

    for axis in axes.flat:
        if axis is not axes[0, 0]:
            axis.set_xticks(x_positions)
            axis.set_xticklabels(labels, rotation=25, ha="right")
        axis.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_markdown(results, output_path):
    lines = [
        "# CartPole evaluation summary",
        "",
        "| Method | Recovery | Recovery [s] | Recovery IAE | Peak error | Final RMSE | Response energy | Solver mean/p95 [ms] | ESS/K |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metrics, _ in results:
        recovery = (
            f'{metrics["recovery_time_s"]:.2f}'
            if metrics["recovery_time_s"] is not None
            else "FAIL"
        )
        lines.append(
            f'| {metrics["display_name"]} '
            f'| {metrics["recovery_success"]} '
            f'| {recovery} '
            f'| {metrics["recovery_weighted_iae"]:.4f} '
            f'| {metrics["peak_disturbance_response_weighted_error"]:.4f} '
            f'| {metrics["final_2s_weighted_rmse"]:.4f} '
            f'| {metrics["disturbance_response_control_energy_n2_s"]:.1f} '
            f'| {metrics["solver_time_mean_ms"]:.3f} / '
            f'{metrics["solver_time_p95_ms"]:.3f} '
            f'| {metrics["ess_ratio_mean"]:.4f} |'
        )
    lines.extend(
        [
            "",
            "실제 plant에만 t=3.0~4.0 s 동안 theta_ddot=+6 rad/s² 외란을 적용했다.",
            "최초 정착과 지속 외란 종료 후 재정착 판정은 각 method의 `metrics.json`에 기록되어 있다.",
            "단일 seed 결과이므로 통계적 우열이 아니라 평가 파이프라인 확인용이다.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_videos(results, result_root):
    for metrics, trajectory in results:
        states = trajectory["states"]
        forces = trajectory["force_n"]
        delta_t = metrics["configuration"]["dt_s"]
        cartpole = CartPole(delta_t=delta_t, visualize=True)
        view_limit = max(6.0, float(np.max(np.abs(states[:, 0]))) + 2.0)
        cartpole.ax.set_xlim(-view_limit, view_limit)

        for state, force in zip(states[1:], forces):
            cartpole.state = state.copy()
            cartpole.append_frame(float(force))

        output_path = result_root / metrics["method"] / "cartpole.mp4"
        cartpole.save_animation(
            str(output_path),
            interval=int(round(delta_t * 1000)),
            movie_writer="ffmpeg",
        )
        plt.close(cartpole.fig)
        print(f"[saved] {output_path}")


def main():
    result_root = (
        Path(__file__).resolve().parents[2] / "result" / "cartpole_eval"
    )
    result_root.mkdir(parents=True, exist_ok=True)
    results = load_results(result_root)
    write_csv(results, result_root / "summary.csv")
    save_comparison_plot(results, result_root / "comparison.png")
    write_markdown(results, result_root / "summary.md")
    save_videos(results, result_root)
    print((result_root / "summary.md").read_text(encoding="utf-8"))
    print(f"[saved] {result_root}")


if __name__ == "__main__":
    main()
