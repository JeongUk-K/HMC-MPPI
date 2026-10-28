"""Shared BARN-map unicycle model and evaluation loop."""

import csv
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from collsion_checker.collision_model import CollisionModel
from model.model_base import ModelBase


DT = 0.1
HORIZON = int(os.environ.get("WMROBOT_HORIZON", "50"))
NUM_SAMPLES = int(os.environ.get("WMROBOT_NUM_SAMPLES", "3000"))
TEMPERATURE = 0.1
SIM_STEPS = int(os.environ.get("WMROBOT_SIM_STEPS", "200"))
SEED = int(os.environ.get("WMROBOT_SEED", "42"))
MAP_ID = int(os.environ.get("WMROBOT_MAP_ID", "0"))
ALL_MAPS = os.environ.get("WMROBOT_ALL_MAPS", "0") == "1"
MAP_COUNT = int(os.environ.get("WMROBOT_MAP_COUNT", "300"))
START_X = float(os.environ.get("WMROBOT_START_X", "0.5"))
SAVE_PLOTS = os.environ.get(
    "WMROBOT_SAVE_PLOTS", "0" if ALL_MAPS else "1"
) == "1"
MAP_RESOLUTION = 0.1
TARGET_STATE = np.array([1.5, 5.0, np.pi / 2.0])
GOAL_TOLERANCE = 0.1
COLLISION_PENALTY = 1.0e8


class UnicycleModel(ModelBase):
    """Point unicycle matching the reference WMRobotMap dynamics."""

    @property
    def action_min(self):
        return jnp.array([0.0, -jnp.pi / 2.0])

    @property
    def action_max(self):
        return jnp.array([1.0, jnp.pi / 2.0])

    def dynamics(self, state, action):
        velocity = jnp.clip(action[0], 0.0, 1.0)
        yaw_rate = jnp.clip(action[1], -jnp.pi / 2.0, jnp.pi / 2.0)
        x, y, yaw = state
        return jnp.array(
            [
                x + DT * velocity * jnp.cos(yaw),
                y + DT * velocity * jnp.sin(yaw),
                yaw + DT * yaw_rate,
            ]
        )

    @staticmethod
    def state_error(state, target_state):
        error = state - target_state
        yaw_error = (error[2] + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
        return error.at[2].set(yaw_error)

    def running_cost(self, state, action, target_state):
        del action
        return jnp.linalg.norm(self.state_error(state, target_state))

    def terminal_cost(self, state, final_target_state):
        return jnp.linalg.norm(self.state_error(state, final_target_state))


class BarnCollisionChecker(CollisionModel):
    """Point collision check matching the reference BARN text-map logic."""

    def check_collision(self, collision_state, collision_context):
        grid = collision_context["grid"]
        resolution = collision_context["resolution"]
        grid_x = jnp.rint(collision_state[..., 0] / resolution).astype(jnp.int32)
        grid_y = jnp.rint(collision_state[..., 1] / resolution).astype(jnp.int32)
        outside_x = (grid_x < 0) | (grid_x >= grid.shape[0])
        below_map = grid_y < 0
        above_map = grid_y >= grid.shape[1]
        safe_x = jnp.clip(grid_x, 0, grid.shape[0] - 1)
        safe_y = jnp.clip(grid_y, 0, grid.shape[1] - 1)
        occupied = grid[safe_x, safe_y] == 10.0

        # As in the reference, the upper map edge is the free-space goal exit.
        return outside_x | below_map | (~above_map & occupied)


def load_barn_map(project_root, map_id):
    if not 0 <= map_id < 300:
        raise ValueError("WMROBOT_MAP_ID must be between 0 and 299")
    map_path = (
        project_root
        / "ref"
        / "BiC-MPPI"
        / "BARN_dataset"
        / "txt_files"
        / f"output_{map_id}.txt"
    )
    if not map_path.is_file():
        raise FileNotFoundError(f"BARN map not found: {map_path}")
    grid = np.loadtxt(map_path, dtype=np.float32)
    if grid.ndim != 2 or grid.size == 0:
        raise ValueError(f"Invalid BARN map: {map_path}")
    return map_path, grid


def wrapped_state_error(states):
    errors = states - TARGET_STATE
    errors[:, 2] = (errors[:, 2] + np.pi) % (2.0 * np.pi) - np.pi
    return errors


def point_is_collision(state, grid):
    grid_x = round(float(state[0]) / MAP_RESOLUTION)
    grid_y = round(float(state[1]) / MAP_RESOLUTION)
    if grid_x < 0 or grid_x >= grid.shape[0] or grid_y < 0:
        return True
    if grid_y >= grid.shape[1]:
        return False
    return bool(grid[grid_x, grid_y] == 10.0)


def calculate_metrics(
    method,
    display_name,
    map_id,
    start_state,
    case_seed,
    states,
    actions,
    solver_times_ms,
    ess_history,
    minimum_costs,
    success,
    collision,
    map_path,
    solver_configuration,
):
    errors = wrapped_state_error(states)
    position_errors = np.linalg.norm(errors[:, :2], axis=1)
    state_errors = np.linalg.norm(errors, axis=1)
    configuration = {
        "map_id": map_id,
        "map_path": str(map_path),
        "map_resolution_m": MAP_RESOLUTION,
        "dt_s": DT,
        "horizon": HORIZON,
        "num_samples": NUM_SAMPLES,
        "temperature": TEMPERATURE,
        "seed": case_seed,
        "max_steps": SIM_STEPS,
        "start_state": start_state.tolist(),
        "target_state": TARGET_STATE.tolist(),
        "goal_tolerance": GOAL_TOLERANCE,
        "collision_penalty": COLLISION_PENALTY,
    }
    configuration.update(solver_configuration)
    return {
        "method": method,
        "display_name": display_name,
        "success": bool(success),
        "collision": bool(collision),
        "steps": int(len(actions)),
        "travel_time_s": float(len(actions) * DT),
        "path_length_m": float(
            np.sum(np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1))
        ),
        "final_position_error_m": float(position_errors[-1]),
        "final_heading_error_rad": float(abs(errors[-1, 2])),
        "final_state_error": float(state_errors[-1]),
        "control_linear_rms_m_s": float(np.sqrt(np.mean(actions[:, 0] ** 2))),
        "control_angular_rms_rad_s": float(np.sqrt(np.mean(actions[:, 1] ** 2))),
        "solver_time_mean_ms": float(np.mean(solver_times_ms)),
        "solver_time_std_ms": float(np.std(solver_times_ms)),
        "solver_time_p95_ms": float(np.percentile(solver_times_ms, 95)),
        "solver_time_max_ms": float(np.max(solver_times_ms)),
        "solver_frequency_hz": float(1000.0 / np.mean(solver_times_ms)),
        "ess_mean": float(np.mean(ess_history)),
        "ess_min": float(np.min(ess_history)),
        "ess_ratio_mean": float(np.mean(ess_history) / NUM_SAMPLES),
        "minimum_predicted_cost": float(np.min(minimum_costs)),
        "final_predicted_cost": float(minimum_costs[-1]),
        "configuration": configuration,
        "timing_note": (
            "JAX compilation is warmed up before measurement. ESS, host "
            "conversion, plant dynamics, plotting, and file I/O are outside "
            "solver timing."
        ),
    }


def save_plot(
    method,
    display_name,
    map_id,
    start_state,
    noise_labels,
    grid,
    times,
    states,
    actions,
    ess_history,
    sample_counts,
    minimum_costs,
    success,
    collision,
    output_path,
):
    fig, axes = plt.subplots(3, 2, figsize=(14, 13))

    map_axis = axes[0, 0]
    map_axis.imshow(
        grid.T,
        origin="lower",
        cmap="gray_r",
        interpolation="nearest",
        extent=(
            -0.5 * MAP_RESOLUTION,
            (grid.shape[0] - 0.5) * MAP_RESOLUTION,
            -0.5 * MAP_RESOLUTION,
            (grid.shape[1] - 0.5) * MAP_RESOLUTION,
        ),
    )
    map_axis.plot(states[:, 0], states[:, 1], color="tab:blue", label="path")
    map_axis.scatter(*start_state[:2], color="tab:green", label="start")
    map_axis.scatter(
        *TARGET_STATE[:2], marker="*", s=150, color="tab:red", label="goal"
    )
    outcome = "success" if success else "collision" if collision else "timeout"
    map_axis.set_title(f"BARN map {map_id}: {outcome}")
    map_axis.set_ylim(-0.1, TARGET_STATE[1] + 0.15)
    map_axis.set_aspect("equal")
    map_axis.set_xlabel("x [m]")
    map_axis.set_ylabel("y [m]")
    map_axis.legend()

    errors = wrapped_state_error(states)
    axes[0, 1].plot(times, states[:, 0], label="x [m]")
    axes[0, 1].plot(times, states[:, 1], label="y [m]")
    axes[0, 1].plot(times, states[:, 2], label="yaw [rad]")
    axes[0, 1].set_title(f"{display_name}: states")
    axes[0, 1].legend()

    action_times = times[1:]
    axes[1, 0].step(action_times, actions[:, 0], where="post", label="v [m/s]")
    axes[1, 0].step(
        action_times, actions[:, 1], where="post", label="omega [rad/s]"
    )
    axes[1, 0].set_title("control inputs")
    axes[1, 0].legend()

    axes[1, 1].plot(
        times, np.linalg.norm(errors[:, :2], axis=1), label="position"
    )
    axes[1, 1].plot(times, np.abs(errors[:, 2]), label="heading")
    axes[1, 1].axhline(
        GOAL_TOLERANCE, color="black", linestyle=":", label="goal tolerance"
    )
    axes[1, 1].set_title("goal error")
    axes[1, 1].legend()

    axes[2, 0].plot(action_times, ess_history / NUM_SAMPLES, color="tab:purple")
    axes[2, 0].set_ylim(0.0, 1.05)
    axes[2, 0].set_title("effective sample size")
    axes[2, 0].set_ylabel("ESS / K")

    for component, label in enumerate(noise_labels):
        axes[2, 1].step(
            action_times,
            sample_counts[:, component] / NUM_SAMPLES,
            where="post",
            label=label,
        )
    axes[2, 1].set_title("sample allocation")
    axes[2, 1].set_ylabel("K_m / K")
    cost_axis = axes[2, 1].twinx()
    cost_axis.plot(
        action_times, minimum_costs, color="black", alpha=0.45, label="min cost"
    )
    cost_axis.set_ylabel("minimum predicted cost")
    lines, labels = axes[2, 1].get_legend_handles_labels()
    cost_lines, cost_labels = cost_axis.get_legend_handles_labels()
    axes[2, 1].legend(lines + cost_lines, labels + cost_labels, loc="best")

    for axis in axes.flat:
        if axis is not map_axis:
            axis.set_xlabel("time [s]")
        axis.grid(True)
    fig.suptitle(
        f"{method} / BARN map {map_id} / start x={start_state[0]:.1f}"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _run_case(
    project_root,
    model,
    solver,
    method,
    display_name,
    noise_labels,
    solver_configuration,
    adaptive,
    map_id,
    start_state,
    case_seed,
):
    map_path, grid = load_barn_map(project_root, map_id)
    method_dir = project_root / "result" / "wmrobot" / method
    if ALL_MAPS:
        start_name = f"start_x_{start_state[0]:.1f}".replace(".", "p")
        result_dir = method_dir / start_name / f"map_{map_id:03d}"
    else:
        result_dir = method_dir / f"map_{map_id:03d}"
    result_dir.mkdir(parents=True, exist_ok=True)

    collision_context = {
        "grid": jnp.asarray(grid),
        "resolution": jnp.asarray(MAP_RESOLUTION),
    }
    ref_trajectory = jnp.tile(jnp.asarray(TARGET_STATE), (HORIZON, 1))
    mean_action_seq = jnp.zeros((HORIZON, 2))
    state = jnp.asarray(start_state)
    proposal_weights = jnp.ones(len(noise_labels)) / len(noise_labels)
    key = jax.random.PRNGKey(case_seed)

    def solve(solve_key, current_state, action_sequence, proposals):
        if adaptive:
            outputs = solver.step(
                solve_key,
                current_state,
                action_sequence,
                ref_trajectory,
                proposals,
                True,
                collision_context,
                True,
            )
            action, action_sequence, proposals, counts, weights, cost = outputs
        else:
            outputs = solver.step(
                solve_key,
                current_state,
                action_sequence,
                ref_trajectory,
                True,
                collision_context,
                True,
            )
            action, action_sequence, weights, cost = outputs
            counts = jnp.array([NUM_SAMPLES])
            proposals = jnp.ones(1)
        return action, action_sequence, proposals, counts, weights, cost

    warm_outputs = solve(
        jax.random.PRNGKey(case_seed + 10000),
        state,
        mean_action_seq,
        proposal_weights,
    )
    jax.block_until_ready(warm_outputs)

    states = [np.asarray(state)]
    actions = []
    solver_times_ms = []
    ess_history = []
    sample_count_history = []
    proposal_weight_history = []
    minimum_cost_history = []
    success = False
    collision = False

    if not ALL_MAPS:
        print(f"=== {display_name} BARN map {map_id} evaluation start ===")
    for step in range(SIM_STEPS):
        key, subkey = jax.random.split(key)
        start = time.perf_counter()
        outputs = solve(subkey, state, mean_action_seq, proposal_weights)
        jax.block_until_ready(outputs)
        solver_times_ms.append((time.perf_counter() - start) * 1000.0)
        (
            action,
            mean_action_seq,
            proposal_weights,
            sample_counts,
            weights,
            minimum_cost,
        ) = outputs

        weights_np = np.asarray(weights)
        ess_history.append(float(1.0 / np.sum(weights_np**2)))
        actions.append(np.asarray(action))
        sample_count_history.append(np.asarray(sample_counts))
        proposal_weight_history.append(np.asarray(proposal_weights))
        minimum_cost_history.append(float(np.asarray(minimum_cost)))

        state = model.dynamics(state, action)
        jax.block_until_ready(state)
        state_np = np.asarray(state)
        states.append(state_np)
        collision = point_is_collision(state_np, grid)
        error = wrapped_state_error(np.asarray([state_np]))[0]
        success = np.linalg.norm(error) < GOAL_TOLERANCE

        if not ALL_MAPS and (step % 20 == 19 or success or collision):
            print(
                f"{step + 1:3d}/{SIM_STEPS} | "
                f"position=({state_np[0]:.2f}, {state_np[1]:.2f}) | "
                f"error={np.linalg.norm(error):.3f} | "
                f"ESS/K={ess_history[-1] / NUM_SAMPLES:.3f} | "
                f"solver={solver_times_ms[-1]:.3f} ms"
            )
        if collision or success:
            break

    states = np.asarray(states)
    actions = np.asarray(actions)
    solver_times_ms = np.asarray(solver_times_ms)
    ess_history = np.asarray(ess_history)
    sample_counts = np.asarray(sample_count_history)
    proposal_weight_history = np.asarray(proposal_weight_history)
    minimum_costs = np.asarray(minimum_cost_history)
    times = np.arange(len(states)) * DT
    errors = wrapped_state_error(states)
    metrics = calculate_metrics(
        method,
        display_name,
        map_id,
        start_state,
        case_seed,
        states,
        actions,
        solver_times_ms,
        ess_history,
        minimum_costs,
        success,
        collision,
        map_path,
        solver_configuration,
    )

    np.savez(
        result_dir / "trajectory.npz",
        time_s=times,
        states=states,
        wrapped_errors=errors,
        actions=actions,
        solver_time_ms=solver_times_ms,
        ess=ess_history,
        ess_ratio=ess_history / NUM_SAMPLES,
        sample_counts=sample_counts,
        proposal_weights=proposal_weight_history,
        minimum_predicted_cost=minimum_costs,
        map_grid=grid,
        map_resolution_m=MAP_RESOLUTION,
        start_state=start_state,
        target_state=TARGET_STATE,
    )
    with (result_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)
    if SAVE_PLOTS:
        save_plot(
            method,
            display_name,
            map_id,
            start_state,
            noise_labels,
            grid,
            times,
            states,
            actions,
            ess_history,
            sample_counts,
            minimum_costs,
            success,
            collision,
            result_dir / "evaluation.png",
        )
    if not ALL_MAPS:
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        print(f"[saved] {result_dir}")
    return metrics, result_dir


def run_evaluation(
    method,
    display_name,
    solver_factory,
    noise_labels,
    solver_configuration,
    adaptive=False,
):
    """Run one selected map, or all 300 maps from both reference starts."""
    if not 1 <= MAP_COUNT <= 300:
        raise ValueError("WMROBOT_MAP_COUNT must be between 1 and 300")

    project_root = Path(__file__).resolve().parents[2]
    model = UnicycleModel()
    collision_checker = BarnCollisionChecker(model, state_indices=(0, 1))
    solver = solver_factory(model, collision_checker)
    if ALL_MAPS:
        cases = [
            (map_id, start_x, SEED + map_id + start_index * 300)
            for start_index, start_x in enumerate((0.5, 2.5))
            for map_id in range(MAP_COUNT)
        ]
    else:
        if not 0 <= MAP_ID < 300:
            raise ValueError("WMROBOT_MAP_ID must be between 0 and 299")
        cases = [(MAP_ID, START_X, SEED)]

    method_dir = project_root / "result" / "wmrobot" / method
    method_dir.mkdir(parents=True, exist_ok=True)
    summary_path = method_dir / "barn_summary.csv"
    summary_fields = (
        "method",
        "map_id",
        "start_x",
        "start_y",
        "success",
        "collision",
        "steps",
        "travel_time_s",
        "path_length_m",
        "final_position_error_m",
        "final_heading_error_rad",
        "final_state_error",
        "solver_time_mean_ms",
        "ess_ratio_mean",
        "result_dir",
    )

    successes = 0
    collisions = 0
    with summary_path.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=summary_fields)
        writer.writeheader()
        for case_index, (map_id, start_x, case_seed) in enumerate(cases, start=1):
            start_state = np.array([start_x, 0.0, np.pi / 2.0])
            metrics, result_dir = _run_case(
                project_root,
                model,
                solver,
                method,
                display_name,
                noise_labels,
                solver_configuration,
                adaptive,
                map_id,
                start_state,
                case_seed,
            )
            successes += int(metrics["success"])
            collisions += int(metrics["collision"])
            writer.writerow(
                {
                    "method": method,
                    "map_id": map_id,
                    "start_x": start_x,
                    "start_y": 0.0,
                    "success": metrics["success"],
                    "collision": metrics["collision"],
                    "steps": metrics["steps"],
                    "travel_time_s": metrics["travel_time_s"],
                    "path_length_m": metrics["path_length_m"],
                    "final_position_error_m": metrics[
                        "final_position_error_m"
                    ],
                    "final_heading_error_rad": metrics[
                        "final_heading_error_rad"
                    ],
                    "final_state_error": metrics["final_state_error"],
                    "solver_time_mean_ms": metrics["solver_time_mean_ms"],
                    "ess_ratio_mean": metrics["ess_ratio_mean"],
                    "result_dir": str(result_dir),
                }
            )
            summary_file.flush()
            if ALL_MAPS:
                outcome = (
                    "success"
                    if metrics["success"]
                    else "collision"
                    if metrics["collision"]
                    else "timeout"
                )
                print(
                    f"[{case_index:3d}/{len(cases)}] "
                    f"start=({start_x:.1f}, 0.0) map={map_id:03d} | "
                    f"{outcome} | steps={metrics['steps']}"
                )

    if ALL_MAPS:
        print(
            f"=== {display_name}: {len(cases)} cases complete | "
            f"success={successes}, collision={collisions}, "
            f"timeout={len(cases) - successes - collisions} ==="
        )
        print(f"[summary] {summary_path}")
