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

from collsion_checker.collision_model import CollisionModel
from model.model_base import ModelBase
from solver.gr_mppi_v2 import GRMPPISolver


METHOD = "gr_mppi_v2"
DISPLAY_NAME = "GR-MPPI adaptive K"
DT = 0.02
HORIZON = 32
NUM_SAMPLES = int(os.environ.get("CR_MPPI_QUAD_NUM_SAMPLES", "8192"))
SIM_STEPS = int(os.environ.get("CR_MPPI_QUAD_SIM_STEPS", "300"))
TEMPERATURE = 0.01
SEED = 42
MASS = 0.027
GRAVITY = 9.81
MAX_THRUST = 0.8
MAX_OMEGA = jnp.array([10.0, 10.0, 3.0])
HOVER_ACTION = 2.0 * MASS * GRAVITY / MAX_THRUST - 1.0
COLLISION_PENALTY = 100.0
SIGMAS = (0.2, 0.5, 1.0)


def quaternion_multiply(left, right):
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return jnp.array([
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ])


def quaternion_rotation(quaternion):
    x, y, z, w = quaternion
    return jnp.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ])


def quaternion_yaw(quaternion):
    quaternion = quaternion / jnp.linalg.norm(quaternion)
    x, y, z, w = quaternion
    return jnp.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


class QuadrotorModel(ModelBase):
    """state=[position(3), quaternion xyzw(4), velocity(3), body rate(3)]."""

    @property
    def action_min(self):
        return -jnp.ones(4)

    @property
    def action_max(self):
        return jnp.ones(4)

    def dynamics(self, state, action):
        action = jnp.clip(action, self.action_min, self.action_max)
        position = state[:3]
        quaternion = state[3:7] / jnp.linalg.norm(state[3:7])
        velocity = state[7:10]
        body_rate = state[10:13]

        thrust = 0.5 * (action[0] + 1.0) * MAX_THRUST
        target_body_rate = action[1:] * MAX_OMEGA
        quaternion_dot = 0.5 * quaternion_multiply(
            quaternion, jnp.concatenate((body_rate, jnp.zeros(1)))
        )
        acceleration = (
            jnp.array([0.0, 0.0, -GRAVITY])
            + quaternion_rotation(quaternion) @ jnp.array([0.0, 0.0, thrust]) / MASS
        )
        next_quaternion = quaternion + DT * quaternion_dot
        next_quaternion = next_quaternion / jnp.linalg.norm(next_quaternion)
        return jnp.concatenate((
            position + DT * velocity,
            next_quaternion,
            velocity + DT * acceleration,
            0.5 * body_rate + 0.5 * target_body_rate,
        ))

    # action에 대한 cost는 안들어가고 position error 항만 들어간다
    def running_cost(self, state, action, target_state):
        position_error = jnp.linalg.norm(state[:3] - target_state[:3])
        velocity_error = jnp.linalg.norm(state[7:10] - target_state[3:6])
        log_position_cost = (
            0.4 * position_error
            + 0.4 * jnp.clip(4.0 * jnp.log1p(position_error), 0.0, 1.0)
            + 0.2 * jnp.clip(8.0 * jnp.log1p(position_error), 0.0, 1.0)
            + 0.1 * jnp.clip(16.0 * jnp.log1p(position_error), 0.0, 1.0)
            + 0.1 * jnp.clip(32.0 * jnp.log1p(position_error), 0.0, 1.0)
        )
        return log_position_cost + 0.05 * velocity_error + 0.2 * jnp.abs(
            quaternion_yaw(state[3:7])
        )


class WorkspaceCollision(CollisionModel):
    """CoVO 예제의 종료 조건인 |x|, |y|, |z| > 3을 충돌로 취급한다."""

    def check_collision(self, position, collision_context):
        limit = collision_context["workspace_limit"]
        return jnp.any(jnp.abs(position) > limit, axis=-1)


def generate_zigzag_trajectory(max_steps, seed):
    """원본 tracking_zigzag와 같은 40-step 3차원 선분 궤적을 만든다."""
    points_per_segment = 40
    segments = max_steps // points_per_segment + 1
    key = jax.random.PRNGKey(seed)
    point_keys = jax.random.split(key, segments + 1)
    angle_keys = jax.random.split(key, segments + 1)
    previous = jax.random.uniform(point_keys[0], (3,), minval=-1.0, maxval=1.0)
    previous = previous / jnp.linalg.norm(previous) * 0.1
    position_segments = []
    velocity_segments = []
    for index in range(segments):
        toward_center = -previous / jnp.linalg.norm(previous)
        delta_theta, delta_phi = jax.random.uniform(
            angle_keys[index + 1], (2,), minval=-jnp.pi / 3.0, maxval=jnp.pi / 3.0
        )
        theta = jnp.arccos(jnp.clip(toward_center[2], -1.0, 1.0)) + delta_theta
        phi = jnp.arctan2(toward_center[1], toward_center[0]) + delta_phi
        direction = jnp.array([
            jnp.sin(theta) * jnp.cos(phi),
            jnp.sin(theta) * jnp.sin(phi),
            jnp.cos(theta),
        ])
        distance = jax.random.uniform(point_keys[index + 1], minval=1.0, maxval=1.5)
        following = previous + distance * direction
        fraction = jnp.arange(points_per_segment, dtype=jnp.float32)[:, None] / points_per_segment
        position_segments.append(previous + fraction * (following - previous))
        velocity_segments.append(jnp.tile(
            ((following - previous) / (points_per_segment + 1) / DT)[None, :],
            (points_per_segment, 1),
        ))
        previous = following
    positions = jnp.concatenate(position_segments)
    positions = positions - positions[0]
    velocities = jnp.concatenate(velocity_segments)
    required = max_steps + HORIZON + 1
    padding = max(0, required - positions.shape[0])
    positions = jnp.pad(positions, ((0, padding), (0, 0)), mode="edge")
    velocities = jnp.pad(velocities, ((0, padding), (0, 0)), mode="edge")
    return positions, velocities


def noisy_measurement(state, key):
    position_key, velocity_key, quaternion_key, omega_key = jax.random.split(key, 4)
    return state.at[:3].add(0.0125 * jax.random.normal(position_key, (3,))).at[7:10].add(
        0.025 * jax.random.normal(velocity_key, (3,))
    ).at[3:7].add(
        0.001 * jax.random.normal(quaternion_key, (4,))
    ).at[10:13].add(
        0.025 * jax.random.normal(omega_key, (3,))
    )


def yaw_history(states):
    q = states[:, 3:7] / np.linalg.norm(states[:, 3:7], axis=1, keepdims=True)
    x, y, z, w = q.T
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def save_plot(result_dir, states, references, actions, ess, counts):
    times = np.arange(len(states)) * DT
    action_times = times[:-1]
    position_error = np.linalg.norm(states[:, :3] - references[:, :3], axis=1)
    velocity_error = np.linalg.norm(states[:, 7:10] - references[:, 3:6], axis=1)
    fig = plt.figure(figsize=(16, 14))
    axis_3d = fig.add_subplot(4, 2, 1, projection="3d")
    axis_3d.plot(*references[:, :3].T, "--", color="gray", label="reference")
    axis_3d.plot(*states[:, :3].T, color="tab:blue", label="actual")
    axis_3d.set_title("3D tracking trajectory")
    axis_3d.legend()
    axes = [fig.add_subplot(4, 2, index) for index in range(2, 9)]
    axes[0].plot(times, position_error)
    axes[0].set_title("Position error [m]")
    for index, name in enumerate("xyz"):
        axes[1].plot(times, states[:, index], label=f"{name}")
        axes[1].plot(times, references[:, index], "--", alpha=0.7, label=f"{name}_ref")
    axes[1].set_title("Position")
    axes[1].legend(ncol=3, fontsize=8)
    axes[2].plot(times, velocity_error)
    axes[2].set_title("Velocity error [m/s]")
    axes[3].plot(times, yaw_history(states))
    axes[3].set_title("Yaw [rad]")
    action_names = ("thrust", "rate_x", "rate_y", "rate_z")
    for index, name in enumerate(action_names):
        axes[4].step(action_times, actions[:, index], where="post", label=name)
    axes[4].set_title("Normalized action")
    axes[4].legend(fontsize=8)
    axes[5].plot(action_times, ess / NUM_SAMPLES)
    axes[5].set_title("ESS / K")
    for component in range(counts.shape[1]):
        axes[6].step(action_times, counts[:, component], where="post", label=f"K{component + 1}")
    axes[6].set_title("Adaptive sample allocation")
    axes[6].legend()
    for axis in [axis_3d, *axes]:
        axis.grid(True, alpha=0.3)
    fig.suptitle(DISPLAY_NAME)
    fig.tight_layout()
    fig.savefig(result_dir / "evaluation.png", dpi=150)
    plt.close(fig)


def main():
    project_root = Path(__file__).resolve().parents[2]
    result_dir = project_root / "result/quadrotor_eval" / METHOD
    result_dir.mkdir(parents=True, exist_ok=True)
    positions, velocities = generate_zigzag_trajectory(SIM_STEPS, SEED)
    model = QuadrotorModel()
    checker = WorkspaceCollision(model, (0, 1, 2))
    collision_context = {"workspace_limit": jnp.array(3.0)}
    solver = GRMPPISolver(
        model=model,
        horizon=HORIZON,
        num_samples=NUM_SAMPLES,
        temperature=TEMPERATURE,
        noise_sigmas=SIGMAS,
        target_weights=(1.0, 1.0, 1.0),
        defensive_ratio=0.2,
        min_samples_per_component=max(1, NUM_SAMPLES // 10),
        collision_checker=checker,
        collision_penalty=COLLISION_PENALTY,
    )

    @jax.jit
    def rollout_prediction(initial_state, action_sequence):
        def step(rollout_state, rollout_action):
            next_state = model.dynamics(rollout_state, rollout_action)
            return next_state, next_state
        _, predicted = jax.lax.scan(step, initial_state, action_sequence)
        return jnp.concatenate((initial_state[None, :], predicted), axis=0)

    state = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    hover = jnp.array([HOVER_ACTION, 0.0, 0.0, 0.0])
    mean_actions = jnp.tile(hover, (HORIZON, 1))
    proposal_weights = jnp.ones(3) / 3.0
    reference = jnp.concatenate((positions[:HORIZON], velocities[:HORIZON]), axis=1)
    warm = solver.step(
        jax.random.PRNGKey(SEED + 100), state, mean_actions, reference,
        proposal_weights, True, collision_context,
    )
    jax.block_until_ready(warm)
    jax.block_until_ready(rollout_prediction(state, mean_actions))

    solver_key = jax.random.PRNGKey(SEED + 1)
    measurement_key = jax.random.PRNGKey(SEED + 2)
    states = [np.asarray(state)]
    predicted_states = []
    actions, solver_times, ess_history, count_history, collision_history = [], [], [], [], []
    print(f"=== {DISPLAY_NAME} quadrotor tracking start ===")
    for step in range(SIM_STEPS):
        reference = jnp.concatenate((
            positions[step:step + HORIZON], velocities[step:step + HORIZON]
        ), axis=1)
        measurement_key, sub_measurement_key = jax.random.split(measurement_key)
        measured_state = noisy_measurement(state, sub_measurement_key)
        solver_key, sub_solver_key = jax.random.split(solver_key)
        start = time.perf_counter()
        outputs = solver.step(
            sub_solver_key, measured_state, mean_actions, reference,
            proposal_weights, True, collision_context,
        )
        jax.block_until_ready(outputs)
        solver_times.append((time.perf_counter() - start) * 1000.0)
        action, mean_actions, proposal_weights, counts, weights = outputs
        optimal_actions = jnp.concatenate((action[None, :], mean_actions[:-1]), axis=0)
        prediction = rollout_prediction(measured_state, optimal_actions)
        jax.block_until_ready(prediction)
        predicted_states.append(np.asarray(prediction))
        mean_actions = mean_actions.at[-1].set(mean_actions[-2])
        state = model.dynamics(state, action)
        jax.block_until_ready(state)
        states.append(np.asarray(state))
        actions.append(np.asarray(action))
        weights_np = np.asarray(weights)
        ess_history.append(float(1.0 / np.sum(weights_np**2)))
        count_history.append(np.asarray(counts))
        collision_history.append(bool(np.asarray(checker.is_collision(state, collision_context))))
        if step % 50 == 49:
            print(f"{step + 1}/{SIM_STEPS} | solver={solver_times[-1]:.3f} ms | K={count_history[-1]}")

    states = np.asarray(states)
    predicted_states = np.asarray(predicted_states)
    actions = np.asarray(actions)
    solver_times = np.asarray(solver_times)
    ess_history = np.asarray(ess_history)
    count_history = np.asarray(count_history)
    collision_history = np.asarray(collision_history)
    references = np.column_stack((np.asarray(positions[:len(states)]), np.asarray(velocities[:len(states)])))
    position_error = np.linalg.norm(states[:, :3] - references[:, :3], axis=1)
    velocity_error = np.linalg.norm(states[:, 7:10] - references[:, 3:6], axis=1)
    yaw = yaw_history(states)
    metrics = {
        "method": METHOD,
        "display_name": DISPLAY_NAME,
        "position_rmse_m": float(np.sqrt(np.mean(position_error**2))),
        "position_mean_error_m": float(position_error.mean()),
        "position_max_error_m": float(position_error.max()),
        "velocity_rmse_mps": float(np.sqrt(np.mean(velocity_error**2))),
        "yaw_rmse_rad": float(np.sqrt(np.mean(yaw**2))),
        "control_total_variation": float(np.abs(np.diff(actions, axis=0)).sum()),
        "collision_steps": int(collision_history.sum()),
        "solver_time_mean_ms": float(solver_times.mean()),
        "solver_time_p95_ms": float(np.percentile(solver_times, 95)),
        "ess_mean": float(ess_history.mean()),
        "ess_ratio_mean": float(ess_history.mean() / NUM_SAMPLES),
        "sample_count_mean": count_history.mean(axis=0).tolist(),
        "sample_count_min": count_history.min(axis=0).tolist(),
        "sample_count_max": count_history.max(axis=0).tolist(),
        "configuration": {
            "task": "tracking_zigzag", "dt_s": DT, "horizon": HORIZON,
            "sim_steps": SIM_STEPS,
            "num_samples": NUM_SAMPLES, "temperature": TEMPERATURE,
            "seed": SEED, "noise_sigmas": SIGMAS,
            "observation_noise": True, "domain_randomization": False,
            "disturbance": "none", "collision_penalty": COLLISION_PENALTY,
            "is_smoke_test": NUM_SAMPLES != 8192 or SIM_STEPS != 300,
        },
        "timing_note": "JIT warm-up excluded; ESS, prediction rollout, plant, metrics, plotting and I/O are outside solver timing.",
    }
    np.savez(
        result_dir / "trajectory.npz", states=states, actions=actions,
        predicted_states=predicted_states, references=references,
        time_s=np.arange(len(states)) * DT, ess=ess_history,
        sample_counts=count_history, collision=collision_history,
        position_error=position_error, velocity_error=velocity_error,
    )
    with (result_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)
    save_plot(result_dir, states, references, actions, ess_history, count_history)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"[saved] {result_dir}")


if __name__ == "__main__":
    main()
