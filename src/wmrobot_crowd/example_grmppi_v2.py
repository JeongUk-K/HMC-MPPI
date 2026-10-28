"""MPPI variants with configurable moving-crowd prediction."""

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
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import numpy as np

from collsion_checker.collision_model import CollisionModel
from model.model_base import ModelBase
from solver.gr_mppi_v2 import GRMPPISolver
from solver.logmppi import LogMPPISolver
from solver.mppi import MPPISolver


DT = 0.1
HORIZON = int(os.environ.get("WMROBOT_CROWD_HORIZON", "50"))
PEDESTRIAN_PREDICTION_HORIZON_S = float(
    os.environ.get("WMROBOT_CROWD_PREDICTION_HORIZON_S", "1.0")
)
PEDESTRIAN_MEASUREMENT_STD_M = float(
    os.environ.get("WMROBOT_CROWD_MEASUREMENT_STD_M", "0.05")
)
PEDESTRIAN_PROCESS_ACCEL_STD_M_S2 = float(
    os.environ.get("WMROBOT_CROWD_PROCESS_ACCEL_STD_M_S2", "0.5")
)
PEDESTRIAN_INITIAL_VELOCITY_STD_M_S = float(
    os.environ.get("WMROBOT_CROWD_INITIAL_VELOCITY_STD_M_S", "1.0")
)
NUM_SAMPLES = int(os.environ.get("WMROBOT_CROWD_NUM_SAMPLES", "3000"))
TEMPERATURE = float(os.environ.get("WMROBOT_CROWD_TEMPERATURE", "1.0"))
SIM_STEPS = int(os.environ.get("WMROBOT_CROWD_SIM_STEPS", "200"))
SAVE_ANIMATION = os.environ.get("WMROBOT_CROWD_SAVE_ANIMATION", "1") == "1"
SEED = int(os.environ.get("WMROBOT_CROWD_SEED", "42"))
DESIRED_SPEED = 0.75
START_STATE = np.array([0.0, 0.0, 0.0, 0.0])
GOAL_STATE = np.array([10.0, 0.0, 0.0, 0.0])
ROBOT_RADIUS = 0.18
PEDESTRIAN_RADIUS = 0.22
SAFE_DISTANCE = ROBOT_RADIUS + PEDESTRIAN_RADIUS
CORRIDOR_HALF_WIDTH = 2.0
COLLISION_PENALTY = 1.0e6
NOISE_SIGMAS = ((0.12, 0.25), (0.25, 0.5), (0.45, 1.0))
MIN_SAMPLES_PER_COMPONENT = max(1, NUM_SAMPLES // 10)
MPPI_NOISE_SIGMA = (0.25, 0.5)
LOG_NOISE_MEAN = 1.023
LOG_NOISE_STD = 0.048**0.5
LOG_NOISE_SCALE = np.exp(LOG_NOISE_MEAN + LOG_NOISE_STD**2)
LOG_BASE_SIGMA = tuple(sigma / LOG_NOISE_SCALE for sigma in MPPI_NOISE_SIGMA)

# [center_x, center_y, amplitude_x, amplitude_y,
#  omega_x, omega_y, phase_x, phase_y]
# Different frequencies and phases create crossing, diagonal, and looping paths.
PEDESTRIAN_PARAMS = np.array(
    [
        [1.5, 0.0, 0.20, 1.55, 0.45, 0.72, 0.0, -1.10],
        [2.6, 0.1, 0.45, 1.45, 0.65, 0.55, 1.0, 0.30],
        [3.7, -0.1, 0.30, 1.60, 0.40, 0.83, 2.0, -2.20],
        [4.8, 0.0, 0.55, 1.35, 0.72, 0.48, -0.5, 1.10],
        [5.9, 0.1, 0.25, 1.65, 0.53, 0.76, 1.7, -0.45],
        [7.0, -0.1, 0.50, 1.40, 0.61, 0.57, -1.5, 2.20],
        [8.1, 0.0, 0.30, 1.60, 0.47, 0.69, 0.8, -2.80],
        [9.0, 0.0, 0.45, 1.45, 0.68, 0.52, 2.4, 0.75],
        [4.8, 0.95, 3.70, 0.30, 0.22, 0.61, -1.2, 0.20],
        [5.2, -0.95, 3.60, 0.35, 0.19, 0.54, 1.8, -0.80],
        [3.2, 0.25, 0.75, 0.80, 0.58, 0.58, 0.1, 1.67],
        [7.4, -0.20, 0.85, 0.70, 0.49, 0.49, -1.0, 0.57],
    ],
    dtype=np.float32,
)
PEDESTRIAN_PARAMS_JAX = jnp.asarray(PEDESTRIAN_PARAMS)


def pedestrian_positions(time_s):
    params = PEDESTRIAN_PARAMS_JAX
    time_s = jnp.asarray(time_s)[..., None]
    x = params[:, 0] + params[:, 2] * jnp.sin(
        params[:, 4] * time_s + params[:, 6]
    )
    y = params[:, 1] + params[:, 3] * jnp.sin(
        params[:, 5] * time_s + params[:, 7]
    )
    return jnp.stack((x, y), axis=-1)


def pedestrian_positions_np(time_s):
    params = PEDESTRIAN_PARAMS
    time_s = np.asarray(time_s)[..., None]
    x = params[:, 0] + params[:, 2] * np.sin(
        params[:, 4] * time_s + params[:, 6]
    )
    y = params[:, 1] + params[:, 3] * np.sin(
        params[:, 5] * time_s + params[:, 7]
    )
    return np.stack((x, y), axis=-1)


def initialize_kalman_filter(measurement):
    mean = np.column_stack((measurement, np.zeros_like(measurement)))
    covariance = np.tile(
        np.diag(
            [
                PEDESTRIAN_MEASUREMENT_STD_M**2,
                PEDESTRIAN_MEASUREMENT_STD_M**2,
                PEDESTRIAN_INITIAL_VELOCITY_STD_M_S**2,
                PEDESTRIAN_INITIAL_VELOCITY_STD_M_S**2,
            ]
        ),
        (len(measurement), 1, 1),
    )
    return mean, covariance


def update_kalman_filter(mean, covariance, measurement):
    transition = np.array(
        [
            [1.0, 0.0, DT, 0.0],
            [0.0, 1.0, 0.0, DT],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    observation = np.eye(2, 4)
    acceleration_input = np.array(
        [[0.5 * DT**2, 0.0], [0.0, 0.5 * DT**2], [DT, 0.0], [0.0, DT]]
    )
    process_covariance = (
        PEDESTRIAN_PROCESS_ACCEL_STD_M_S2**2
        * acceleration_input
        @ acceleration_input.T
    )
    measurement_covariance = PEDESTRIAN_MEASUREMENT_STD_M**2 * np.eye(2)
    predicted_mean = mean @ transition.T
    predicted_covariance = (
        transition @ covariance @ transition.T + process_covariance
    )
    identity = np.eye(4)

    for pedestrian in range(len(measurement)):
        covariance_xy = predicted_covariance[pedestrian] @ observation.T
        innovation_covariance = (
            observation @ covariance_xy + measurement_covariance
        )
        gain = covariance_xy @ np.linalg.inv(innovation_covariance)
        predicted_mean[pedestrian] += gain @ (
            measurement[pedestrian]
            - observation @ predicted_mean[pedestrian]
        )
        correction = identity - gain @ observation
        predicted_covariance[pedestrian] = (
            correction
            @ predicted_covariance[pedestrian]
            @ correction.T
            + gain @ measurement_covariance @ gain.T
        )
    return predicted_mean, predicted_covariance


def predict_crowd(estimated_positions, estimated_velocities, elapsed_time):
    prediction_time = jnp.clip(
        jnp.asarray(elapsed_time),
        0.0,
        PEDESTRIAN_PREDICTION_HORIZON_S,
    )
    return (
        estimated_positions
        + prediction_time[..., None, None] * estimated_velocities
    )


class CrowdUnicycleModel(ModelBase):
    """State is [x, y, yaw, time]."""

    @property
    def action_min(self):
        return jnp.array([0.0, -1.5])

    @property
    def action_max(self):
        return jnp.array([1.2, 1.5])

    def dynamics(self, state, action):
        velocity = jnp.clip(action[0], 0.0, 1.2)
        yaw_rate = jnp.clip(action[1], -1.5, 1.5)
        x, y, yaw, time_s = state
        return jnp.array(
            [
                x + DT * velocity * jnp.cos(yaw),
                y + DT * velocity * jnp.sin(yaw),
                yaw + DT * yaw_rate,
                time_s + DT,
            ]
        )

    @staticmethod
    def _yaw_error(yaw, target_yaw):
        return (yaw - target_yaw + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

    def running_cost(self, state, action, target_state):
        position = state[:2]
        predicted_crowd = target_state[4:].reshape((-1, 2))
        distances = jnp.linalg.norm(
            predicted_crowd - position,
            axis=1,
        )
        tracking_cost = (
            0.8 * (state[0] - target_state[0]) ** 2
            + 4.0 * state[1] ** 2
            + 0.4 * self._yaw_error(state[2], 0.0) ** 2
        )
        social_cost = 3.0 * jnp.sum(jnp.exp(-0.5 * (distances / 0.65) ** 2))
        control_cost = 0.03 * action[1] ** 2
        return tracking_cost + social_cost + control_cost

    def terminal_cost(self, state, final_target_state):
        return 3.0 * (
            (state[0] - final_target_state[0]) ** 2
            + 4.0 * state[1] ** 2
            + 0.4 * self._yaw_error(state[2], 0.0) ** 2
        )


class MovingCrowdCollisionChecker(CollisionModel):
    def check_collision(self, collision_state, collision_context):
        position = collision_state[..., :2]
        state_time = collision_state[..., 2]
        predicted_crowd = predict_crowd(
            collision_context["estimated_positions"],
            collision_context["estimated_velocities"],
            state_time - collision_context["observation_time"],
        )
        crowd_collision = jnp.any(
            jnp.linalg.norm(
                predicted_crowd - position[..., None, :],
                axis=-1,
            )
            < SAFE_DISTANCE,
            axis=-1,
        )
        wall_collision = jnp.abs(position[..., 1]) > CORRIDOR_HALF_WIDTH
        return crowd_collision | wall_collision


def make_reference(state, estimated_positions, estimated_velocities):
    steps = jnp.arange(1, HORIZON + 1)
    future_time = state[3] + steps * DT
    future_x = jnp.minimum(
        state[0] + DESIRED_SPEED * steps * DT,
        GOAL_STATE[0],
    )
    route = jnp.column_stack(
        (future_x, jnp.zeros(HORIZON), jnp.zeros(HORIZON), future_time)
    )
    predicted_crowd = predict_crowd(
        estimated_positions,
        estimated_velocities,
        jnp.arange(HORIZON) * DT,
    )
    return jnp.concatenate(
        (route, predicted_crowd.reshape((HORIZON, -1))), axis=1
    )


def minimum_clearance(state):
    crowd = pedestrian_positions_np(float(state[3]))
    return float(np.min(np.linalg.norm(crowd - state[:2], axis=1)))


def self_check():
    if PEDESTRIAN_PREDICTION_HORIZON_S < 0.0:
        raise ValueError("WMROBOT_CROWD_PREDICTION_HORIZON_S must be >= 0")
    if PEDESTRIAN_MEASUREMENT_STD_M <= 0.0:
        raise ValueError("WMROBOT_CROWD_MEASUREMENT_STD_M must be > 0")
    if PEDESTRIAN_PROCESS_ACCEL_STD_M_S2 < 0.0:
        raise ValueError("WMROBOT_CROWD_PROCESS_ACCEL_STD_M_S2 must be >= 0")
    if PEDESTRIAN_INITIAL_VELOCITY_STD_M_S <= 0.0:
        raise ValueError(
            "WMROBOT_CROWD_INITIAL_VELOCITY_STD_M_S must be > 0"
        )
    positions_now = pedestrian_positions_np(0.0)
    positions_later = pedestrian_positions_np(1.0)
    assert positions_now.shape == (len(PEDESTRIAN_PARAMS), 2)
    assert not np.allclose(positions_now, positions_later)
    assert minimum_clearance(START_STATE) > SAFE_DISTANCE
    mean, covariance = initialize_kalman_filter(positions_now)
    mean, covariance = update_kalman_filter(
        mean, covariance, pedestrian_positions_np(DT)
    )
    assert mean.shape == (len(PEDESTRIAN_PARAMS), 4)
    assert covariance.shape == (len(PEDESTRIAN_PARAMS), 4, 4)
    assert np.all(np.isfinite(mean)) and np.all(np.isfinite(covariance))
    estimated_positions = mean[:, :2]
    estimated_velocities = mean[:, 2:]
    reference = np.asarray(
        make_reference(
            jnp.asarray(START_STATE),
            jnp.asarray(estimated_positions),
            jnp.asarray(estimated_velocities),
        )
    )
    last_prediction_time = min(
        (HORIZON - 1) * DT,
        PEDESTRIAN_PREDICTION_HORIZON_S,
    )
    assert np.allclose(
        reference[-1, 4:].reshape((-1, 2)),
        estimated_positions + last_prediction_time * estimated_velocities,
    )


def make_solver(method, model, collision_checker):
    common = {
        "model": model,
        "horizon": HORIZON,
        "num_samples": NUM_SAMPLES,
        "temperature": TEMPERATURE,
        "collision_checker": collision_checker,
        "collision_penalty": COLLISION_PENALTY,
    }
    if method == "mppi":
        return (
            MPPISolver(noise_sigma=MPPI_NOISE_SIGMA, **common),
            "MPPI",
            ("sigma=(0.25, 0.5)",),
            {"noise_sigma": list(MPPI_NOISE_SIGMA)},
            False,
        )
    if method == "logmppi":
        return (
            LogMPPISolver(
                noise_sigma=LOG_BASE_SIGMA,
                log_noise_mean=LOG_NOISE_MEAN,
                log_noise_std=LOG_NOISE_STD,
                **common,
            ),
            "Log-MPPI",
            ("NLN RMS=(0.25, 0.5)",),
            {
                "reference_noise_rms": list(MPPI_NOISE_SIGMA),
                "base_noise_sigma": list(LOG_BASE_SIGMA),
                "log_noise_mean": LOG_NOISE_MEAN,
                "log_noise_std": LOG_NOISE_STD,
            },
            False,
        )
    if method == "gr_mppi_v2":
        return (
            GRMPPISolver(
                noise_sigmas=NOISE_SIGMAS,
                target_weights=(0.0, 1.0, 0.0),
                defensive_ratio=0.2,
                min_samples_per_component=MIN_SAMPLES_PER_COMPONENT,
                **common,
            ),
            "GR-MPPI adaptive K",
            tuple(
                f"sigma=({sigma[0]:g}, {sigma[1]:g})"
                for sigma in NOISE_SIGMAS
            ),
            {
                "noise_sigmas": [list(sigma) for sigma in NOISE_SIGMAS],
                "defensive_ratio": 0.2,
                "min_samples_per_component": MIN_SAMPLES_PER_COMPONENT,
            },
            True,
        )
    raise ValueError(f"Unknown method: {method}")


def save_plot(
    display_name,
    noise_labels,
    states,
    actions,
    solver_times_ms,
    ess_history,
    sample_counts,
    clearances,
    output_path,
):
    times = states[:, 3]
    action_times = times[1:]
    fig, axes = plt.subplots(3, 2, figsize=(14, 13))

    path_axis = axes[0, 0]
    crowd_times = np.linspace(0.0, times[-1], 300)
    crowd_paths = np.asarray([pedestrian_positions_np(t) for t in crowd_times])
    for pedestrian in range(len(PEDESTRIAN_PARAMS)):
        path_axis.plot(
            crowd_paths[:, pedestrian, 0],
            crowd_paths[:, pedestrian, 1],
            color="tab:orange",
            alpha=0.35,
        )
    path_axis.axhline(0.0, color="black", linestyle="--", label="straight reference")
    path_axis.plot(states[:, 0], states[:, 1], color="tab:blue", linewidth=2, label="robot")
    path_axis.scatter(*START_STATE[:2], color="tab:green", label="start")
    path_axis.scatter(*GOAL_STATE[:2], marker="*", s=150, color="tab:red", label="goal")
    path_axis.set(xlim=(-0.4, 10.4), ylim=(-2.1, 2.1), xlabel="x [m]", ylabel="y [m]")
    path_axis.set_aspect("equal")
    path_axis.set_title(f"{display_name}: straight route through moving crowd")
    path_axis.legend(loc="upper center", ncol=4)

    axes[0, 1].plot(times, states[:, 0], label="x [m]")
    axes[0, 1].plot(times, states[:, 1], label="y [m]")
    axes[0, 1].plot(times, states[:, 2], label="yaw [rad]")
    axes[0, 1].set_title("robot states")
    axes[0, 1].legend()

    axes[1, 0].step(action_times, actions[:, 0], where="post", label="v [m/s]")
    axes[1, 0].step(action_times, actions[:, 1], where="post", label="omega [rad/s]")
    axes[1, 0].set_title("control inputs")
    axes[1, 0].legend()

    axes[1, 1].plot(times, clearances, color="tab:red")
    axes[1, 1].axhline(SAFE_DISTANCE, color="black", linestyle=":", label="collision distance")
    axes[1, 1].set_title("nearest pedestrian distance")
    axes[1, 1].set_ylabel("distance [m]")
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
    axes[2, 1].set_title("adaptive sample allocation")
    axes[2, 1].set_ylabel("K_m / K")
    axes[2, 1].legend()

    for axis in axes.flat:
        axis.grid(True, alpha=0.3)
        if axis is not path_axis:
            axis.set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_animation(display_name, states, output_path):
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.set(xlim=(-0.4, 10.4), ylim=(-2.1, 2.1), xlabel="x [m]", ylabel="y [m]")
    axis.set_aspect("equal")
    axis.grid(True, alpha=0.2)
    axis.axhline(0.0, color="black", linestyle="--", label="straight reference")
    axis.axhline(CORRIDOR_HALF_WIDTH, color="black", linewidth=2)
    axis.axhline(-CORRIDOR_HALF_WIDTH, color="black", linewidth=2)
    axis.scatter(*GOAL_STATE[:2], marker="*", s=180, color="tab:red", label="goal")
    path_line, = axis.plot([], [], color="tab:blue", linewidth=2, label="robot path")
    robot = Circle((0.0, 0.0), ROBOT_RADIUS, color="tab:blue")
    axis.add_patch(robot)
    pedestrians = []
    for _ in PEDESTRIAN_PARAMS:
        pedestrian = Circle((0.0, 0.0), PEDESTRIAN_RADIUS, color="tab:orange")
        pedestrians.append(pedestrian)
        axis.add_patch(pedestrian)
    axis.legend(loc="upper center", ncol=3)

    def update(frame):
        state = states[frame]
        crowd = pedestrian_positions_np(float(state[3]))
        robot.center = tuple(state[:2])
        path_line.set_data(states[: frame + 1, 0], states[: frame + 1, 1])
        for pedestrian, position in zip(pedestrians, crowd):
            pedestrian.center = tuple(position)
        axis.set_title(
            f"{display_name} | t={state[3]:.1f}s | "
            f"clearance={minimum_clearance(state):.2f}m"
        )
        return (path_line, robot, *pedestrians)

    animation = FuncAnimation(
        figure,
        update,
        frames=len(states),
        interval=DT * 1000.0,
        blit=False,
    )
    animation.save(output_path, writer="ffmpeg", fps=round(1.0 / DT), dpi=120)
    plt.close(figure)


def main(method="gr_mppi_v2"):
    self_check()
    result_dir = (
        Path(__file__).resolve().parents[2]
        / "result"
        / "wmrobot_crowd"
        / method
    )
    result_dir.mkdir(parents=True, exist_ok=True)

    model = CrowdUnicycleModel()
    collision_checker = MovingCrowdCollisionChecker(
        model, state_indices=(0, 1, 3)
    )
    solver, display_name, noise_labels, solver_configuration, adaptive = (
        make_solver(method, model, collision_checker)
    )
    state = jnp.asarray(START_STATE)
    mean_actions = jnp.zeros((HORIZON, 2))
    proposal_weights = jnp.ones(len(noise_labels)) / len(noise_labels)
    key = jax.random.PRNGKey(SEED)
    measurement_rng = np.random.default_rng(SEED + 1)
    initial_measurement = pedestrian_positions_np(0.0) + measurement_rng.normal(
        0.0,
        PEDESTRIAN_MEASUREMENT_STD_M,
        (len(PEDESTRIAN_PARAMS), 2),
    )
    kalman_mean, kalman_covariance = initialize_kalman_filter(
        initial_measurement
    )

    def solve(
        solve_key,
        current_state,
        action_sequence,
        proposals,
        estimated_positions,
        estimated_velocities,
    ):
        reference = make_reference(
            current_state, estimated_positions, estimated_velocities
        )
        collision_context = {
            "observation_time": current_state[3],
            "estimated_positions": estimated_positions,
            "estimated_velocities": estimated_velocities,
        }
        if adaptive:
            return solver.step(
                solve_key,
                current_state,
                action_sequence,
                reference,
                proposals,
                True,
                collision_context,
                True,
            )
        action, action_sequence, weights, cost = solver.step(
            solve_key,
            current_state,
            action_sequence,
            reference,
            True,
            collision_context,
            True,
        )
        return (
            action,
            action_sequence,
            jnp.ones(1),
            jnp.array([NUM_SAMPLES]),
            weights,
            cost,
        )

    warm_outputs = solve(
        jax.random.PRNGKey(SEED + 10000),
        state,
        mean_actions,
        proposal_weights,
        jnp.asarray(kalman_mean[:, :2]),
        jnp.asarray(kalman_mean[:, 2:]),
    )
    jax.block_until_ready(warm_outputs)

    states = [np.asarray(state)]
    actions = []
    solver_times_ms = []
    ess_history = []
    sample_count_history = []
    proposal_weight_history = []
    minimum_cost_history = []
    clearances = [minimum_clearance(START_STATE)]
    pedestrian_measurements = [initial_measurement]
    pedestrian_estimated_positions = [kalman_mean[:, :2].copy()]
    pedestrian_estimated_velocities = [kalman_mean[:, 2:].copy()]
    success = False
    collision = False

    print(
        f"=== {display_name} moving-crowd evaluation start "
        f"(prediction={PEDESTRIAN_PREDICTION_HORIZON_S:g}s) ==="
    )
    for step in range(SIM_STEPS):
        key, subkey = jax.random.split(key)
        start = time.perf_counter()
        outputs = solve(
            subkey,
            state,
            mean_actions,
            proposal_weights,
            jnp.asarray(kalman_mean[:, :2]),
            jnp.asarray(kalman_mean[:, 2:]),
        )
        jax.block_until_ready(outputs)
        solver_times_ms.append((time.perf_counter() - start) * 1000.0)
        (
            action,
            mean_actions,
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
        measurement = pedestrian_positions_np(
            float(state_np[3])
        ) + measurement_rng.normal(
            0.0,
            PEDESTRIAN_MEASUREMENT_STD_M,
            (len(PEDESTRIAN_PARAMS), 2),
        )
        kalman_mean, kalman_covariance = update_kalman_filter(
            kalman_mean, kalman_covariance, measurement
        )
        pedestrian_measurements.append(measurement)
        pedestrian_estimated_positions.append(kalman_mean[:, :2].copy())
        pedestrian_estimated_velocities.append(kalman_mean[:, 2:].copy())
        clearance = minimum_clearance(state_np)
        clearances.append(clearance)
        collision = (
            clearance < SAFE_DISTANCE
            or abs(state_np[1]) > CORRIDOR_HALF_WIDTH
        )
        position_error = np.linalg.norm(state_np[:2] - GOAL_STATE[:2])
        yaw_error = abs((state_np[2] + np.pi) % (2.0 * np.pi) - np.pi)
        success = position_error < 0.25 and yaw_error < 0.35

        if step % 20 == 19 or success or collision:
            print(
                f"{step + 1:3d}/{SIM_STEPS} | "
                f"position=({state_np[0]:.2f}, {state_np[1]:+.2f}) | "
                f"clearance={clearance:.2f}m | "
                f"ESS/K={ess_history[-1] / NUM_SAMPLES:.3f}"
            )
        if success or collision:
            break

    states = np.asarray(states)
    actions = np.asarray(actions)
    solver_times_ms = np.asarray(solver_times_ms)
    ess_history = np.asarray(ess_history)
    sample_counts = np.asarray(sample_count_history)
    proposal_weights_np = np.asarray(proposal_weight_history)
    minimum_costs = np.asarray(minimum_cost_history)
    clearances = np.asarray(clearances)
    final_position_error = np.linalg.norm(states[-1, :2] - GOAL_STATE[:2])
    metrics = {
        "method": method,
        "display_name": display_name,
        "scenario": "straight route with configurable crowd prediction",
        "success": bool(success),
        "collision": bool(collision),
        "steps": int(len(actions)),
        "travel_time_s": float(len(actions) * DT),
        "path_length_m": float(
            np.sum(np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1))
        ),
        "final_position_error_m": float(final_position_error),
        "minimum_pedestrian_clearance_m": float(np.min(clearances)),
        "solver_time_mean_ms": float(np.mean(solver_times_ms)),
        "solver_time_p95_ms": float(np.percentile(solver_times_ms, 95)),
        "ess_mean": float(np.mean(ess_history)),
        "ess_ratio_mean": float(np.mean(ess_history) / NUM_SAMPLES),
        "configuration": {
            "dt_s": DT,
            "horizon": HORIZON,
            "num_samples": NUM_SAMPLES,
            "temperature": TEMPERATURE,
            "seed": SEED,
            "num_pedestrians": len(PEDESTRIAN_PARAMS),
            "robot_radius_m": ROBOT_RADIUS,
            "pedestrian_radius_m": PEDESTRIAN_RADIUS,
            "desired_speed_m_s": DESIRED_SPEED,
            "pedestrian_prediction": (
                "constant-velocity Kalman filter; frozen after prediction horizon"
            ),
            "pedestrian_prediction_horizon_s": (
                PEDESTRIAN_PREDICTION_HORIZON_S
            ),
            "controller_horizon_s": HORIZON * DT,
            "pedestrian_measurement_std_m": PEDESTRIAN_MEASUREMENT_STD_M,
            "pedestrian_process_accel_std_m_s2": (
                PEDESTRIAN_PROCESS_ACCEL_STD_M_S2
            ),
            "pedestrian_initial_velocity_std_m_s": (
                PEDESTRIAN_INITIAL_VELOCITY_STD_M_S
            ),
            **solver_configuration,
        },
    }

    np.savez(
        result_dir / "trajectory.npz",
        states=states,
        actions=actions,
        solver_time_ms=solver_times_ms,
        ess=ess_history,
        sample_counts=sample_counts,
        proposal_weights=proposal_weights_np,
        minimum_predicted_cost=minimum_costs,
        pedestrian_clearance_m=clearances,
        pedestrian_params=PEDESTRIAN_PARAMS,
        pedestrian_measurements=np.asarray(pedestrian_measurements),
        pedestrian_estimated_positions=np.asarray(
            pedestrian_estimated_positions
        ),
        pedestrian_estimated_velocities=np.asarray(
            pedestrian_estimated_velocities
        ),
    )
    with (result_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)
    save_plot(
        display_name,
        noise_labels,
        states,
        actions,
        solver_times_ms,
        ess_history,
        sample_counts,
        clearances,
        result_dir / "evaluation.png",
    )
    if SAVE_ANIMATION:
        save_animation(display_name, states, result_dir / "animation.mp4")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"[saved] {result_dir}")


if __name__ == "__main__":
    main()
