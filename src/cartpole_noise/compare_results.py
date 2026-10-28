import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
    "initial_settling_time_s",
    "recovery_success",
    "recovery_time_s",
    "recovery_weighted_iae",
    "disturbance_response_total_weighted_iae",
    "peak_disturbance_response_weighted_error",
    "final_2s_weighted_rmse",
    "max_abs_cart_position_m",
    "max_abs_pole_rate_rad_s",
    "control_rms_n",
    "control_energy_n2_s",
    "disturbance_response_control_energy_n2_s",
    "control_max_abs_n",
    "control_total_variation_n",
    "solver_best_cost_initial",
    "solver_best_cost_pre_impulse",
    "solver_best_cost_at_impulse",
    "solver_best_cost_final",
    "solver_best_cost_min",
)


def load_results(result_root: Path):
    results = []
    for method in METHOD_ORDER:
        metrics_path = result_root / method / "metrics.json"
        trajectory_path = result_root / method / "trajectory.npz"
        if not metrics_path.exists() or not trajectory_path.exists():
            raise FileNotFoundError(
                f"Missing noise evaluation output for {method}: run cartpole_noise.sh"
            )
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        results.append((metrics, np.load(trajectory_path)))
    return results


def write_csv(results, output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for metrics, _ in results:
            writer.writerow({field: metrics.get(field) for field in CSV_FIELDS})


def _annotate(axis, bars, values, fmt="{:.2f}") -> None:
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def save_comparison_plot(results, output_path: Path) -> None:
    metrics = [item[0] for item in results]
    labels = [item["display_name"] for item in metrics]
    positions = np.arange(len(labels))
    colors = plt.cm.tab10(np.arange(len(labels)))
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))

    for label, (_, trajectory), color in zip(labels, results, colors):
        axes[0, 0].plot(
            trajectory["time_s"],
            trajectory["weighted_error"],
            label=label,
            color=color,
        )
    axes[0, 0].axvline(
        metrics[0]["disturbance_start_time_s"],
        color="black",
        linestyle="--",
        label="impulse",
    )
    axes[0, 0].set_title("Weighted state error after impulse")
    axes[0, 0].set_xlabel("time [s]")
    axes[0, 0].set_ylabel(r"$\sqrt{e^T Q e}$")
    axes[0, 0].legend(fontsize=8)

    max_recovery = (
        metrics[0]["configuration"]["duration_s"]
        - metrics[0]["disturbance_start_time_s"]
    )
    recovery_times = np.array(
        [
            item["recovery_time_s"]
            if item["recovery_time_s"] is not None
            else max_recovery
            for item in metrics
        ]
    )
    bars = axes[0, 1].bar(positions, recovery_times, color=colors)
    for bar, item in zip(bars, metrics):
        if not item["recovery_success"]:
            bar.set_hatch("//")
            axis_x = bar.get_x() + bar.get_width() / 2
            axes[0, 1].text(axis_x, bar.get_height(), "FAIL", ha="center", va="bottom")
    axes[0, 1].set_title("Recovery time (hatched = failed)")
    axes[0, 1].set_ylabel("time [s]")

    recovery_iae = np.array([item["recovery_weighted_iae"] for item in metrics])
    bars = axes[1, 0].bar(positions, recovery_iae, color=colors)
    _annotate(axes[1, 0], bars, recovery_iae)
    axes[1, 0].set_title("Weighted IAE until recovery")
    axes[1, 0].set_ylabel("weighted error·s")

    for label, (_, trajectory), color in zip(labels, results, colors):
        axes[1, 1].plot(
            trajectory["solver_cost_time_s"],
            trajectory["solver_best_cost"],
            label=label,
            color=color,
        )
    axes[1, 1].axvline(
        metrics[0]["disturbance_start_time_s"],
        color="black",
        linestyle="--",
        label="impulse",
    )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Best sampled rollout cost")
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].set_ylabel("cost")

    final_rmse = np.array([item["final_2s_weighted_rmse"] for item in metrics])
    bars = axes[2, 0].bar(positions, final_rmse, color=colors)
    _annotate(axes[2, 0], bars, final_rmse)
    axes[2, 0].set_title("Final 2 s weighted RMSE")
    axes[2, 0].set_ylabel("weighted error")

    response_energy = np.array(
        [item["disturbance_response_control_energy_n2_s"] for item in metrics]
    )
    bars = axes[2, 1].bar(positions, response_energy, color=colors)
    _annotate(axes[2, 1], bars, response_energy, "{:.0f}")
    axes[2, 1].set_title("Post-impulse control energy")
    axes[2, 1].set_ylabel(r"$\sum u^2 \Delta t$ [N$^2$s]")

    for axis in axes.flat:
        axis.grid(True, alpha=0.3)
        if axis is not axes[0, 0] and axis is not axes[1, 1]:
            axis.set_xticks(positions)
            axis.set_xticklabels(labels, rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_markdown(results, output_path: Path) -> None:
    lines = [
        "# CartPole impulse-noise evaluation summary",
        "",
        "| Method | Recovery | Recovery [s] | Recovery IAE | Peak error | Final RMSE | Cost initial/pre/final | Response energy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
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
            f'| {metrics["solver_best_cost_initial"]:.1f} / '
            f'{metrics["solver_best_cost_pre_impulse"]:.1f} / '
            f'{metrics["solver_best_cost_final"]:.1f} '
            f'| {metrics["disturbance_response_control_energy_n2_s"]:.1f} |'
        )
    lines.extend(
        [
            "",
            "실제 plant에만 t=3.0 s에서 theta_dot=+6 rad/s 순간 임펄스를 적용했다.",
            "정착 판정은 각 method의 `metrics.json`에 기록되어 있다.",
            "단일 seed 결과이므로 통계적 우열이 아니라 평가 파이프라인 확인용이다.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    result_root = Path(__file__).resolve().parents[2] / "result" / "cartpole_noise"
    result_root.mkdir(parents=True, exist_ok=True)
    results = load_results(result_root)
    write_csv(results, result_root / "summary.csv")
    save_comparison_plot(results, result_root / "comparison.png")
    write_markdown(results, result_root / "summary.md")
    print((result_root / "summary.md").read_text(encoding="utf-8"))
    print(f"[saved] {result_root}")


if __name__ == "__main__":
    main()
