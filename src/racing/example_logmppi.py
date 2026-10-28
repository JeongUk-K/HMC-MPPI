import csv
import json
import os
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt

from collsion_checker.collision_model import CollisionModel
from model.model_base import ModelBase
from solver.logmppi import LogMPPISolver


METHOD = "log_mppi"
DISPLAY_NAME = "Log-MPPI"
DT = 0.1
HORIZON = 20
NUM_SAMPLES = int(os.environ.get("CR_MPPI_RACING_NUM_SAMPLES", "4000"))
SIM_STEPS = int(os.environ.get("CR_MPPI_RACING_SIM_STEPS", "500"))
TEMPERATURE = 1.0
SEED = 42
MAP_SIZE = 80.0
CELL_SIZE = 0.1
LANE_WIDTH = 6.5
COLLISION_PENALTY = 10000.0
DYNAMIC_OBSTACLE_EVENTS = ((969, 10.0, 1.1), (2062, 8.0, 1.1), (3166, 6.0, 1.1))
REFERENCE_SIGMA = (0.5, 0.1)
LOG_NOISE_MEAN = 1.023
LOG_NOISE_STD = 0.048**0.5
LOG_NOISE_RMS = np.exp(LOG_NOISE_MEAN + LOG_NOISE_STD**2)
NOISE_SIGMA = tuple(value / LOG_NOISE_RMS for value in REFERENCE_SIGMA)


def load_track(project_root):
    csv_path = (
        project_root
        / "ref/mppi_playground/src/envs/circuit_generator/circuit.csv"
    )
    with csv_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    points = np.array(
        [[float(row["x_m"]), float(row["y_m"])] for row in rows]
    )
    points -= points.mean(axis=0)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(segment_lengths)]
    distances = np.linspace(0.0, cumulative[-1], int(cumulative[-1] / 0.1) + 1)
    path = np.column_stack(
        [np.interp(distances, cumulative, points[:, axis]) for axis in range(2)]
    )
    initial_angle = np.arctan2(*(path[0] - path[-1])[::-1])
    angles = np.r_[initial_angle, np.arctan2(np.diff(path[:, 1]), np.diff(path[:, 0]))]
    return np.column_stack((path, angles)).astype(np.float32)


def make_map(path):
    cells = int(round(MAP_SIZE / CELL_SIZE))
    origin = np.array([cells // 2, cells // 2])
    lane_seed = np.ones((cells, cells), dtype=np.uint8)
    path_cells = np.rint(path[:, :2] / CELL_SIZE + origin).astype(int)
    valid = np.all((path_cells >= 0) & (path_cells < cells), axis=1)
    lane_seed[path_cells[valid, 0], path_cells[valid, 1]] = 0
    lane_map = (
        distance_transform_edt(lane_seed)
        > (LANE_WIDTH * 0.8 / 2.0) / CELL_SIZE
    )

    obstacle_map = np.zeros((cells, cells), dtype=bool)
    circles = []
    rng = np.random.default_rng(SEED)
    while len(circles) < 50:
        center = rng.uniform(-35.0, 35.0, size=2)
        radius = float(rng.uniform(0.9, 1.2))
        if any(np.linalg.norm(center - old_center) <= radius + old_radius
               for old_center, old_radius in circles):
            continue
        circles.append((center, radius))
        center_cell = np.rint(center / CELL_SIZE + origin).astype(int)
        radius_cell = int(np.ceil(radius / CELL_SIZE))
        x0, x1 = np.clip(
            [center_cell[0] - radius_cell, center_cell[0] + radius_cell + 1],
            0,
            cells,
        )
        y0, y1 = np.clip(
            [center_cell[1] - radius_cell, center_cell[1] + radius_cell + 1],
            0,
            cells,
        )
        xx, yy = np.ogrid[x0:x1, y0:y1]
        obstacle_map[x0:x1, y0:y1] |= (
            (xx - center_cell[0]) ** 2 + (yy - center_cell[1]) ** 2
            <= radius_cell**2
        )

    occupancy = np.maximum(lane_map, obstacle_map).astype(np.float32)
    context = {
        "occupancy": jnp.asarray(occupancy),
        "origin": jnp.asarray(origin, dtype=jnp.float32),
        "resolution": jnp.asarray(CELL_SIZE, dtype=jnp.float32),
    }
    return occupancy, np.asarray([(x, y, r) for (x, y), r in circles]), context


def add_circle_to_occupancy(occupancy, center, radius):
    updated = occupancy.copy()
    origin = np.asarray(updated.shape) // 2
    center_cell = np.rint(center / CELL_SIZE + origin).astype(int)
    radius_cell = int(np.ceil(radius / CELL_SIZE))
    x0, x1 = np.clip(
        [center_cell[0] - radius_cell, center_cell[0] + radius_cell + 1],
        0,
        updated.shape[0],
    )
    y0, y1 = np.clip(
        [center_cell[1] - radius_cell, center_cell[1] + radius_cell + 1],
        0,
        updated.shape[1],
    )
    xx, yy = np.ogrid[x0:x1, y0:y1]
    updated[x0:x1, y0:y1] = np.maximum(
        updated[x0:x1, y0:y1],
        (xx - center_cell[0]) ** 2 + (yy - center_cell[1]) ** 2
        <= radius_cell**2,
    )
    return updated


class RacingModel(ModelBase):
    """state=[x, y, yaw, velocity, previous_accel, previous_steer]."""

    @property
    def action_min(self):
        return jnp.array([-2.0, -0.25])

    @property
    def action_max(self):
        return jnp.array([2.0, 0.25])

    def dynamics(self, state, action):
        action = jnp.clip(action, self.action_min, self.action_max)
        x, y, yaw, velocity = state[:4]
        accel, steer = action
        next_state = jnp.array(
            [
                x + velocity * jnp.cos(yaw) * DT,
                y + velocity * jnp.sin(yaw) * DT,
                yaw + velocity * jnp.tan(steer) * DT,
                velocity + accel * DT,
                accel,
                steer,
            ]
        )
        next_state = next_state.at[0:2].set(jnp.clip(next_state[0:2], -40.0, 40.0))
        next_state = next_state.at[2].set(
            (next_state[2] + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
        )
        return next_state.at[3].set(jnp.clip(next_state[3], -8.0, 8.0))

    def running_cost(self, state, action, target_state):
        dx = state[0] - target_state[0]
        dy = state[1] - target_state[1]
        yaw_ref = target_state[2]
        contour = jnp.sin(yaw_ref) * dx - jnp.cos(yaw_ref) * dy
        lag = -jnp.cos(yaw_ref) * dx - jnp.sin(yaw_ref) * dy
        return (
            2.0 * contour**2
            + 3.0 * lag**2
            + 2.0 * (state[3] - target_state[3]) ** 2
            + 0.01 * jnp.sum(action**2)
            + 0.5 * jnp.sum((action - state[4:6]) ** 2)
        )


class OccupancyCollision(CollisionModel):
    def check_collision(self, position, collision_context):
        cells = jnp.rint(
            position / collision_context["resolution"]
            + collision_context["origin"]
        ).astype(jnp.int32)
        shape = jnp.asarray(collision_context["occupancy"].shape)
        out_of_bounds = jnp.any((cells < 0) | (cells >= shape), axis=-1)
        safe = jnp.clip(cells, 0, shape - 1)
        occupied = collision_context["occupancy"][safe[..., 0], safe[..., 1]] > 0.5
        return out_of_bounds | occupied


def reference_trajectory(state, path, current_index):
    distances = np.linalg.norm(path[:, :2] - np.asarray(state[:2]), axis=1)
    nearest = max(current_index, int(np.argmin(distances)))
    travel = 3.0 + 0.85 * (np.arange(HORIZON) + 1)
    raw_indices = nearest + np.rint(travel / 0.1).astype(int)
    indices = np.clip(raw_indices, 0, len(path) - 1)
    reference = np.zeros((HORIZON, 6), dtype=np.float32)
    reference[:, :3] = path[indices]
    reference[:, 3] = np.where(raw_indices < len(path), 8.0, 0.0)
    return jnp.asarray(reference), nearest


def trajectory_errors(states, path):
    contour, lag, speed, indices = [], [], [], []
    current_index = 0
    for state in states:
        nearest = current_index + int(
            np.argmin(np.linalg.norm(path[current_index:, :2] - state[:2], axis=1))
        )
        current_index = min(nearest, len(path) - 1)
        target = path[current_index]
        dx, dy = state[:2] - target[:2]
        contour.append(np.sin(target[2]) * dx - np.cos(target[2]) * dy)
        lag.append(-np.cos(target[2]) * dx - np.sin(target[2]) * dy)
        speed.append(state[3] - (0.0 if current_index == len(path) - 1 else 8.0))
        indices.append(current_index)
    return map(np.asarray, (contour, lag, speed, indices))


def save_plot(result_dir, path, circles, states, actions, ess, counts, appearance_steps):
    contour, lag, speed_error, _ = trajectory_errors(states, path)
    times = np.arange(len(states)) * DT
    action_times = times[1:]
    right = path[:, :2] + 0.5 * LANE_WIDTH * np.column_stack(
        (np.cos(path[:, 2] - np.pi / 2), np.sin(path[:, 2] - np.pi / 2))
    )
    left = path[:, :2] + 0.5 * LANE_WIDTH * np.column_stack(
        (np.cos(path[:, 2] + np.pi / 2), np.sin(path[:, 2] + np.pi / 2))
    )
    fig, axes = plt.subplots(3, 2, figsize=(15, 14))
    axes[0, 0].plot(path[:, 0], path[:, 1], "--", color="gray")
    axes[0, 0].plot(right[:, 0], right[:, 1], color="green", linewidth=0.8)
    axes[0, 0].plot(left[:, 0], left[:, 1], color="green", linewidth=0.8)
    dynamic_start = len(circles) - len(appearance_steps)
    for index, (x, y, radius) in enumerate(circles):
        color = "tab:red" if index >= dynamic_start else "gray"
        axes[0, 0].add_patch(plt.Circle((x, y), radius, color=color))
    axes[0, 0].plot(states[:, 0], states[:, 1], color="tab:blue")
    axes[0, 0].set_aspect("equal")
    axes[0, 0].set_title("Driven trajectory")
    axes[0, 1].plot(times, contour, label="contour")
    axes[0, 1].plot(times, lag, label="lag")
    axes[0, 1].set_title("Path errors [m]")
    axes[0, 1].legend()
    axes[1, 0].plot(times, states[:, 3], label="velocity")
    axes[1, 0].plot(times, speed_error, label="speed error")
    axes[1, 0].legend()
    axes[1, 1].step(action_times, actions[:, 0], where="post", label="accel")
    axes[1, 1].step(action_times, actions[:, 1], where="post", label="steer")
    axes[1, 1].legend()
    axes[2, 0].plot(action_times, ess)
    axes[2, 0].set_title("Effective sample size")
    axes[2, 1].axis("off")
    axes[2, 1].text(
        0.5, 0.5,
        "Normal-Lognormal noise\nRMS matched to sigma [0.5, 0.1]",
        ha="center", va="center", transform=axes[2, 1].transAxes,
    )
    for appearance_step in appearance_steps:
        for axis in (axes[0, 1], axes[1, 0], axes[1, 1], axes[2, 0]):
            axis.axvline(appearance_step * DT, color="tab:red", linestyle="--", linewidth=1)
    for axis in axes.flat:
        axis.grid(True, alpha=0.3)
    fig.suptitle(DISPLAY_NAME)
    fig.tight_layout()
    fig.savefig(result_dir / "evaluation.png", dpi=150)
    plt.close(fig)


def main():
    project_root = Path(__file__).resolve().parents[2]
    result_dir = project_root / "result/racing" / METHOD
    result_dir.mkdir(parents=True, exist_ok=True)
    path = load_track(project_root)
    occupancy, circles, collision_context = make_map(path)
    static_circle_count = len(circles)
    model = RacingModel()

    @jax.jit
    def rollout_prediction(initial_state, action_sequence):
        def step(rollout_state, rollout_action):
            next_state = model.dynamics(rollout_state, rollout_action)
            return next_state, next_state

        _, predicted = jax.lax.scan(step, initial_state, action_sequence)
        return jnp.concatenate((initial_state[None, :], predicted), axis=0)

    checker = OccupancyCollision(model, (0, 1))
    solver = LogMPPISolver(
        model=model,
        horizon=HORIZON,
        num_samples=NUM_SAMPLES,
        temperature=TEMPERATURE,
        noise_sigma=NOISE_SIGMA,
        log_noise_mean=LOG_NOISE_MEAN,
        log_noise_std=LOG_NOISE_STD,
        collision_checker=checker,
        collision_penalty=COLLISION_PENALTY,
    )
    state = jnp.array([path[0, 0], path[0, 1], path[0, 2], 0.0, 0.0, 0.0])
    mean_actions = jnp.zeros((HORIZON, 2))
    reference, current_index = reference_trajectory(state, path, 0)
    key = jax.random.PRNGKey(SEED)

    warm = solver.step(
        jax.random.PRNGKey(SEED + 1), state, mean_actions, reference,
        True, collision_context,
    )
    jax.block_until_ready(warm)
    jax.block_until_ready(rollout_prediction(state, mean_actions))

    states = [np.asarray(state)]
    predicted_states = []
    actions, solver_times, ess_history, collision_history = [], [], [], []
    dynamic_events = []
    next_dynamic_event = 0
    goal_reached = False
    print(f"=== {DISPLAY_NAME} racing evaluation start ===")
    for step in range(SIM_STEPS):
        reference, current_index = reference_trajectory(state, path, current_index)
        while (
            next_dynamic_event < len(DYNAMIC_OBSTACLE_EVENTS)
            and current_index >= DYNAMIC_OBSTACLE_EVENTS[next_dynamic_event][0]
        ):
            trigger_index, ahead_distance, radius = DYNAMIC_OBSTACLE_EVENTS[next_dynamic_event]
            obstacle_index = min(
                trigger_index + int(round(ahead_distance / CELL_SIZE)),
                len(path) - 1,
            )
            center = path[obstacle_index, :2]
            occupancy = add_circle_to_occupancy(occupancy, center, radius)
            circles = np.vstack((circles, [center[0], center[1], radius]))
            collision_context = {
                **collision_context,
                "occupancy": jnp.asarray(occupancy),
            }
            jax.block_until_ready(collision_context["occupancy"])
            dynamic_events.append(
                {
                    "appearance_step": step,
                    "appearance_time_s": step * DT,
                    "trigger_path_index": trigger_index,
                    "actual_trigger_path_index": current_index,
                    "obstacle_path_index": obstacle_index,
                    "ahead_distance_m": ahead_distance,
                    "center_xy_m": center.tolist(),
                    "radius_m": radius,
                }
            )
            print(f"[DYNAMIC OBSTACLE] t={step * DT:.1f}s, path[{obstacle_index}], {ahead_distance:.0f}m ahead")
            next_dynamic_event += 1
        key, subkey = jax.random.split(key)
        start = time.perf_counter()
        outputs = solver.step(
            subkey, state, mean_actions, reference, True, collision_context,
        )
        jax.block_until_ready(outputs)
        solver_times.append((time.perf_counter() - start) * 1000.0)
        action, mean_actions, weights = outputs
        optimal_actions = jnp.concatenate(
            (action[None, :], mean_actions[:-1]), axis=0
        )
        prediction = rollout_prediction(state, optimal_actions)
        jax.block_until_ready(prediction)
        predicted_states.append(np.asarray(prediction))
        state = model.dynamics(state, action)
        jax.block_until_ready(state)
        states.append(np.asarray(state))
        actions.append(np.asarray(action))
        weights_np = np.asarray(weights)
        ess_history.append(float(1.0 / np.sum(weights_np**2)))
        collision_history.append(bool(np.asarray(checker.is_collision(state, collision_context))))
        if np.linalg.norm(np.asarray(state[:2]) - path[-1, :2]) < 1.0:
            goal_reached = True
            break
        if step % 50 == 49:
            print(f"{step + 1}/{SIM_STEPS} | progress={100 * current_index / (len(path) - 1):.1f}% | solver={solver_times[-1]:.2f} ms")

    states = np.asarray(states)
    predicted_states = np.asarray(predicted_states)
    actions = np.asarray(actions)
    solver_times = np.asarray(solver_times)
    ess_history = np.asarray(ess_history)
    count_history = np.empty((len(actions), 0), dtype=np.int32)
    collision_history = np.asarray(collision_history)
    appearance_steps = np.asarray(
        [event["appearance_step"] for event in dynamic_events], dtype=np.int32
    )
    dynamic_circles = circles[static_circle_count:]
    contour, lag, speed_error, path_indices = trajectory_errors(states, path)
    metrics = {
        "method": METHOD,
        "display_name": DISPLAY_NAME,
        "goal_reached": goal_reached,
        "goal_time_s": float((len(states) - 1) * DT) if goal_reached else None,
        "path_progress_percent": float(100.0 * path_indices[-1] / (len(path) - 1)),
        "collision_steps": int(collision_history.sum()),
        "collision_rate": float(collision_history.mean()),
        "dynamic_obstacles": dynamic_events,
        "contour_rmse_m": float(np.sqrt(np.mean(contour**2))),
        "lag_rmse_m": float(np.sqrt(np.mean(lag**2))),
        "speed_rmse_mps": float(np.sqrt(np.mean(speed_error**2))),
        "accel_rms_mps2": float(np.sqrt(np.mean(actions[:, 0] ** 2))),
        "steer_rms_rad": float(np.sqrt(np.mean(actions[:, 1] ** 2))),
        "control_total_variation": float(np.abs(np.diff(actions, axis=0)).sum()),
        "solver_time_mean_ms": float(solver_times.mean()),
        "solver_time_p95_ms": float(np.percentile(solver_times, 95)),
        "ess_mean": float(ess_history.mean()),
        "ess_ratio_mean": float(ess_history.mean() / NUM_SAMPLES),
        "configuration": {
            "dt_s": DT, "horizon": HORIZON, "num_samples": NUM_SAMPLES,
            "temperature": TEMPERATURE, "seed": SEED,
            "reference_sigma": REFERENCE_SIGMA, "noise_sigma": NOISE_SIGMA,
            "log_noise_mean": LOG_NOISE_MEAN, "log_noise_std": LOG_NOISE_STD,
            "collision_penalty": COLLISION_PENALTY,
        },
        "timing_note": "JIT warm-up excluded; ESS, plant, metrics, plotting and I/O are outside solver timing.",
    }
    np.savez(
        result_dir / "trajectory.npz", states=states, actions=actions,
        predicted_states=predicted_states,
        time_s=np.arange(len(states)) * DT, ess=ess_history,
        sample_counts=count_history, collision=collision_history,
        path=path, circles=circles, occupancy=occupancy,
        static_circle_count=static_circle_count,
        dynamic_circles=dynamic_circles,
        dynamic_appearance_steps=appearance_steps,
        contour_error=contour, lag_error=lag, speed_error=speed_error,
    )
    with (result_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)
    save_plot(
        result_dir, path, circles, states, actions, ess_history,
        count_history, appearance_steps,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"[saved] {result_dir}")


if __name__ == "__main__":
    main()
