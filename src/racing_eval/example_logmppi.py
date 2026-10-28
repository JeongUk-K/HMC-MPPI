"""Sudden-obstacle racing scenario for Log-MPPI.

The vehicle cruises on a straight reference at 8 m/s.  At t=5 s an obstacle
appears 20 m ahead and the local planner switches to an adjacent-lane nominal.
At t=8.5 s it switches back to the original line; t>=10 s is used to measure
settling behavior.
"""

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
from solver.logmppi import LogMPPISolver


METHOD = "log_mppi"
DISPLAY_NAME = "Log-MPPI"
ADAPTIVE_SAMPLING = False
DT = 0.1
HORIZON = int(os.environ.get("CR_MPPI_RACING_EVAL_HORIZON", "30"))
NUM_SAMPLES = int(os.environ.get("CR_MPPI_RACING_EVAL_NUM_SAMPLES", "4000"))
SIM_STEPS = int(os.environ.get("CR_MPPI_RACING_EVAL_SIM_STEPS", "200"))
TEMPERATURE = float(os.environ.get("CR_MPPI_RACING_EVAL_TEMPERATURE", "10.0"))
SEED = int(os.environ.get("CR_MPPI_RACING_EVAL_SEED", "42"))

REFERENCE_SPEED = 8.0
LANE_WIDTH = 6.5
VEHICLE_RADIUS = 0.35
OBSTACLE_CENTER = np.array([60.0, 0.0], dtype=np.float32)
OBSTACLE_RADIUS = 1.25
DETOUR_OFFSET = 2.2
EVENT_STEP = 50
DETOUR_END_STEP = 85
SETTLING_START_STEP = 100
COLLISION_PENALTY = 10000.0

# Same initial per-action RMS is used by all three methods.
REFERENCE_SIGMA = np.array([0.5, 0.1], dtype=np.float32)
LOG_NOISE_MEAN = 1.023
LOG_NOISE_STD = 0.048**0.5
LOG_NOISE_RMS = np.exp(LOG_NOISE_MEAN + LOG_NOISE_STD**2)
LOG_NOISE_SIGMA = tuple((REFERENCE_SIGMA / LOG_NOISE_RMS).tolist())

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
        next_state = next_state.at[2].set(
            (next_state[2] + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
        )
        return next_state.at[3].set(jnp.clip(next_state[3], 0.0, 8.0))

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

    def terminal_cost(self, state, final_target_state):
        return (
            4.0 * (state[1] - final_target_state[1]) ** 2
            + 4.0 * (state[3] - final_target_state[3]) ** 2
        )


class SuddenObstacleCollision(CollisionModel):
    def check_collision(self, position, collision_context):
        obstacle_distance = jnp.linalg.norm(
            position - collision_context["obstacle_center"], axis=-1
        )
        obstacle_collision = collision_context["active"] & (
            obstacle_distance
            <= collision_context["obstacle_radius"]
            + collision_context["vehicle_radius"]
        )
        lane_collision = jnp.abs(position[..., 1]) >= (
            collision_context["lane_half_width"]
            - collision_context["vehicle_radius"]
        )
        return obstacle_collision | lane_collision


def reference_trajectory(state, lane_offset):
    steps = jnp.arange(1, HORIZON + 1, dtype=state.dtype)
    reference = jnp.zeros((HORIZON, 6), dtype=state.dtype)
    x = state[0] + REFERENCE_SPEED * DT * steps
    reference = reference.at[:, 0].set(x)
    reference = reference.at[:, 1].set(lane_offset)
    return reference.at[:, 3].set(REFERENCE_SPEED)


def collision_context(active):
    return {
        "active": jnp.asarray(active),
        "obstacle_center": jnp.asarray(OBSTACLE_CENTER),
        "obstacle_radius": jnp.asarray(OBSTACLE_RADIUS),
        "vehicle_radius": jnp.asarray(VEHICLE_RADIUS),
        "lane_half_width": jnp.asarray(LANE_WIDTH / 2.0),
    }


def make_solver(model, checker):
    return LogMPPISolver(
        model=model,
        horizon=HORIZON,
        num_samples=NUM_SAMPLES,
        temperature=TEMPERATURE,
        noise_sigma=LOG_NOISE_SIGMA,
        log_noise_mean=LOG_NOISE_MEAN,
        log_noise_std=LOG_NOISE_STD,
        collision_checker=checker,
        collision_penalty=COLLISION_PENALTY,
    )


def solver_step(solver, key, state, mean_actions, reference, context):
    action, mean_actions, weights, minimum_cost = solver.step(
        key, state, mean_actions, reference, True, context, True
    )
    return action, mean_actions, weights, minimum_cost

def first_sustained_recovery(states, start_step, threshold=0.25, dwell_steps=20):
    within_band = np.abs(states[:, 1]) <= threshold
    for step in range(start_step, len(states) - dwell_steps + 1):
        if np.all(within_band[step : step + dwell_steps]):
            return step
    return None


def evaluate(states, actions, collisions, ess, costs, solver_times):
    combined_radius = OBSTACLE_RADIUS + VEHICLE_RADIUS
    active_states = states[EVENT_STEP:]
    clearance = (
        np.linalg.norm(active_states[:, :2] - OBSTACLE_CENTER, axis=1)
        - combined_radius
    )
    passed = np.flatnonzero(states[:, 0] > OBSTACLE_CENTER[0] + combined_radius)
    passed = passed[passed >= EVENT_STEP]
    pass_step = int(passed[0]) if len(passed) else None
    recovery_step = (
        first_sustained_recovery(states, pass_step) if pass_step is not None else None
    )
    response = np.flatnonzero(np.abs(actions[EVENT_STEP:, 1]) > 0.02)
    response_step = EVENT_STEP + int(response[0]) if len(response) else None
    settling_actions = actions[SETTLING_START_STEP:]
    settling_states = states[SETTLING_START_STEP:]
    action_delta = np.diff(settling_actions, axis=0)
    saturated = np.any(
        np.isclose(np.abs(actions), np.array([2.0, 0.25]), atol=1e-3), axis=1
    )
    return {
        "collision_steps": int(np.sum(collisions)),
        "collision_rate": float(np.mean(collisions)),
        "minimum_obstacle_clearance_m": float(np.min(clearance)),
        "obstacle_pass_time_s": None if pass_step is None else pass_step * DT,
        "response_time_after_event_s": (
            None if response_step is None else (response_step - EVENT_STEP) * DT
        ),
        "recovery_time_after_pass_s": (
            None if recovery_step is None else (recovery_step - pass_step) * DT
        ),
        "settling_contour_mean_abs_m": float(np.mean(np.abs(settling_states[:, 1]))),
        "settling_contour_std_m": float(np.std(settling_states[:, 1])),
        "settling_accel_std_mps2": float(np.std(settling_actions[:, 0])),
        "settling_steer_std_rad": float(np.std(settling_actions[:, 1])),
        "settling_accel_jerk_rms": float(np.sqrt(np.mean(action_delta[:, 0] ** 2))),
        "settling_steer_jerk_rms": float(np.sqrt(np.mean(action_delta[:, 1] ** 2))),
        "action_saturation_rate": float(np.mean(saturated)),
        "ess_mean": float(np.mean(ess)),
        "ess_ratio_mean": float(np.mean(ess) / NUM_SAMPLES),
        "minimum_cost_mean": float(np.mean(costs)),
        "solver_time_mean_ms": float(np.mean(solver_times)),
        "solver_time_p95_ms": float(np.percentile(solver_times, 95)),
    }


def save_plot(result_dir, display_name, states, actions, ess, costs, counts, sampling_rms):
    state_times = np.arange(len(states)) * DT
    action_times = np.arange(len(actions)) * DT
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))

    axes[0, 0].axhspan(-LANE_WIDTH / 2, LANE_WIDTH / 2, color="green", alpha=0.08)
    axes[0, 0].axhline(-LANE_WIDTH / 2, color="green")
    axes[0, 0].axhline(LANE_WIDTH / 2, color="green")
    axes[0, 0].axhline(0.0, color="gray", linestyle=":", label="original nominal")
    axes[0, 0].plot(
        [REFERENCE_SPEED * EVENT_STEP * DT, REFERENCE_SPEED * DETOUR_END_STEP * DT],
        [DETOUR_OFFSET, DETOUR_OFFSET],
        "--",
        color="black",
        label="emergency nominal",
    )
    axes[0, 0].add_patch(
        plt.Circle(OBSTACLE_CENTER, OBSTACLE_RADIUS, color="tab:red", alpha=0.7)
    )
    axes[0, 0].plot(states[:, 0], states[:, 1], color="tab:blue", label="vehicle")
    axes[0, 0].set_ylim(-LANE_WIDTH / 2 - 0.8, LANE_WIDTH / 2 + 0.8)
    axes[0, 0].set_title("Sudden-obstacle trajectory")
    axes[0, 0].legend()

    axes[0, 1].plot(state_times, states[:, 1], label="lateral position")
    axes[0, 1].axhspan(-0.25, 0.25, color="gray", alpha=0.15, label="settled band")
    axes[0, 1].set_title("Lane change and recovery")
    axes[0, 1].legend()

    axes[1, 0].step(action_times, actions[:, 0], where="post", label="accel")
    axes[1, 0].step(action_times, actions[:, 1], where="post", label="steer")
    axes[1, 0].set_title("Control input")
    axes[1, 0].legend()
    axes[1, 1].plot(action_times, ess / NUM_SAMPLES)
    axes[1, 1].set_title("ESS / K")

    axes[2, 0].plot(action_times, costs)
    axes[2, 0].set_title("Minimum sampled trajectory cost")
    axes[2, 1].plot(action_times, sampling_rms[:, 0], label="accel RMS")
    axes[2, 1].plot(action_times, sampling_rms[:, 1], label="steer RMS")
    if counts.shape[1]:
        for component in range(counts.shape[1]):
            axes[2, 1].plot(
                action_times,
                counts[:, component] / NUM_SAMPLES,
                linestyle="--",
                alpha=0.7,
                label=f"K{component + 1}/K",
            )
    axes[2, 1].set_title("Exploration scale and allocation")
    axes[2, 1].legend(ncol=2, fontsize=8)

    for axis in axes.flat:
        axis.axvline(EVENT_STEP * DT, color="tab:red", linestyle="--", linewidth=1)
        axis.axvline(
            DETOUR_END_STEP * DT, color="tab:blue", linestyle="--", linewidth=1
        )
        axis.axvline(
            SETTLING_START_STEP * DT, color="black", linestyle=":", linewidth=1
        )
        axis.grid(True, alpha=0.3)
        axis.set_xlabel("time [s]")
    axes[0, 0].set_xlabel("x [m]")
    axes[0, 0].set_ylabel("y [m] (vertical scale enlarged)")
    fig.suptitle(display_name)
    fig.tight_layout()
    fig.savefig(result_dir / "evaluation.png", dpi=160)
    plt.close(fig)


def main():
    if SIM_STEPS <= SETTLING_START_STEP + 1:
        raise ValueError(
            f"SIM_STEPS must exceed {SETTLING_START_STEP + 1} to evaluate settling"
        )

    project_root = Path(__file__).resolve().parents[2]
    result_dir = project_root / "result/racing_eval" / METHOD
    result_dir.mkdir(parents=True, exist_ok=True)
    model = RacingModel()
    checker = SuddenObstacleCollision(model, (0, 1))
    solver = make_solver(model, checker)
    state = jnp.array([0.0, 0.0, 0.0, REFERENCE_SPEED, 0.0, 0.0])
    mean_actions = jnp.zeros((HORIZON, 2))

    key = jax.random.PRNGKey(SEED)

    warm = solver_step(
        solver,
        jax.random.PRNGKey(SEED + 1),
        state,
        mean_actions,
        reference_trajectory(state, 0.0),
        collision_context(False),
    )
    jax.block_until_ready(warm)

    states = [np.asarray(state)]
    actions = []
    collisions = []
    ess_history = []
    cost_history = []
    count_history = []
    proposal_history = []
    solver_times = []
    print(f"=== {DISPLAY_NAME} sudden-obstacle evaluation ===")

    for step in range(SIM_STEPS):
        active = step >= EVENT_STEP
        lane_offset = DETOUR_OFFSET if EVENT_STEP <= step < DETOUR_END_STEP else 0.0
        context = collision_context(active)
        key, subkey = jax.random.split(key)
        start = time.perf_counter()
        outputs = solver_step(
            solver,
            subkey,
            state,
            mean_actions,
            reference_trajectory(state, lane_offset),
            context,
        )
        jax.block_until_ready(outputs)
        solver_times.append((time.perf_counter() - start) * 1000.0)
        action, mean_actions, weights, minimum_cost = outputs

        state = model.dynamics(state, action)
        jax.block_until_ready(state)
        states.append(np.asarray(state))
        actions.append(np.asarray(action))
        collisions.append(bool(np.asarray(checker.is_collision(state, context))))
        weights = np.asarray(weights)
        ess_history.append(float(1.0 / np.sum(weights**2)))
        cost_history.append(float(np.asarray(minimum_cost)))
        count_history.append(np.asarray(counts))
        proposal_history.append(np.asarray(next_proposal))

        if step == EVENT_STEP:
            distance = float(np.linalg.norm(np.asarray(state[:2]) - OBSTACLE_CENTER))
            print(f"[EVENT] obstacle appeared at t={step * DT:.1f}s, distance={distance:.2f}m")
        if step == DETOUR_END_STEP:
            print(f"[EVENT] nominal returned to center at t={step * DT:.1f}s")
        if step % 50 == 49:
            print(
                f"{step + 1}/{SIM_STEPS} | x={float(state[0]):.1f}m "
                f"| y={float(state[1]):+.2f}m | solver={solver_times[-1]:.2f}ms"
            )

    states = np.asarray(states)
    actions = np.asarray(actions)
    collisions = np.asarray(collisions)
    ess_history = np.asarray(ess_history)
    cost_history = np.asarray(cost_history)
    solver_times = np.asarray(solver_times)
    count_history = np.empty((len(actions), 0), dtype=np.int32)
    proposal_history = np.empty((len(actions), 0), dtype=np.float32)
    sampling_rms = np.broadcast_to(REFERENCE_SIGMA, (len(actions), 2)).copy()
    allocation_metrics = None

    metrics = {
        "method": METHOD,
        "display_name": DISPLAY_NAME,
        **evaluate(
            states, actions, collisions, ess_history, cost_history, solver_times
        ),
        "sample_allocation_ratio": allocation_metrics,
        "configuration": {
            "dt_s": DT,
            "horizon": HORIZON,
            "num_samples": NUM_SAMPLES,
            "temperature": TEMPERATURE,
            "sim_steps": SIM_STEPS,
            "seed": SEED,
            "event_time_s": EVENT_STEP * DT,
            "detour_end_time_s": DETOUR_END_STEP * DT,
            "settling_start_time_s": SETTLING_START_STEP * DT,
            "reference_speed_mps": REFERENCE_SPEED,
            "detour_offset_m": DETOUR_OFFSET,
            "reference_sigma": REFERENCE_SIGMA.tolist(),
            "noise_sigma": list(LOG_NOISE_SIGMA),
            "log_noise_mean": LOG_NOISE_MEAN,
            "log_noise_std": LOG_NOISE_STD,
            "collision_penalty": COLLISION_PENALTY,
        },
        "timing_note": "JIT warm-up excluded; metrics and plant update are outside solver timing.",
    }
    np.savez(
        result_dir / "trajectory.npz",
        states=states,
        actions=actions,
        time_s=np.arange(len(states)) * DT,
        action_time_s=np.arange(len(actions)) * DT,
        collision=collisions,
        ess=ess_history,
        minimum_cost=cost_history,
        sample_counts=count_history,
        proposal_weights=proposal_history,
        sampling_rms=sampling_rms,
        obstacle_center=OBSTACLE_CENTER,
        obstacle_radius=OBSTACLE_RADIUS,
    )
    with (result_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)
    save_plot(
        result_dir,
        DISPLAY_NAME,
        states,
        actions,
        ess_history,
        cost_history,
        count_history,
        sampling_rms,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"[saved] {result_dir}")


if __name__ == "__main__":
    main()

