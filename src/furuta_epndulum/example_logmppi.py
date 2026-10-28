import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from model.model_base import ModelBase
from solver.logmppi import LogMPPISolver
from src.furuta_epndulum.diagnostics import normalized_ess, save_diagnostics


DT = 0.01
HORIZON = int(os.environ.get("CR_MPPI_FURUTA_HORIZON", "70"))
NUM_SAMPLES = int(os.environ.get("CR_MPPI_FURUTA_NUM_SAMPLES", "1024"))
SIM_STEPS = int(os.environ.get("CR_MPPI_FURUTA_SIM_STEPS", "400"))
SAVE_ANIMATION = os.environ.get("CR_MPPI_FURUTA_SAVE_ANIMATION", "1") == "1"
TEMPERATURE = float(os.environ.get("CR_MPPI_FURUTA_TEMPERATURE", "1.0"))
REFERENCE_SIGMA = 0.005**0.5
LOG_NOISE_MEAN = 1.023
LOG_NOISE_STD = 0.048**0.5
LOG_NOISE_RMS = np.exp(LOG_NOISE_MEAN + LOG_NOISE_STD**2)
NOISE_SIGMA = REFERENCE_SIGMA / LOG_NOISE_RMS
SEED = 42


def wrap_angle(angle: jnp.ndarray) -> jnp.ndarray:
    return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


class FurutaPendulumModel(ModelBase):
    """State=[arm angle, pendulum angle, arm rate, pendulum rate].

    Both target angles are zero: the rotary arm is at its home angle and the
    pendulum is upright. The only actuator is the rotary-arm torque.
    """

    def __init__(
        self,
        dt: float = DT,
        arm_length: float = 0.1035,
        pendulum_length: float = 0.191,
        pendulum_com: float = 0.0955,
        pendulum_mass: float = 0.192,
        arm_inertia: float = 2.7e-3,
        pendulum_inertia: float = 2.5e-3,
        arm_damping: float = 2.7e-3,
        pendulum_damping: float = 493.0e-6,
        max_torque: float = 0.45,
    ) -> None:
        self.dt = dt
        self.arm_length = arm_length
        self.pendulum_length = pendulum_length
        self.pendulum_mass = pendulum_mass
        self.pendulum_com = pendulum_com
        self.pendulum_inertia = pendulum_inertia
        self.arm_inertia = arm_inertia
        self.arm_damping = arm_damping
        self.pendulum_damping = pendulum_damping
        self.max_torque = max_torque

    @property
    def action_min(self) -> jnp.ndarray:
        return jnp.array([-self.max_torque])

    @property
    def action_max(self) -> jnp.ndarray:
        return jnp.array([self.max_torque])

    def _derivative(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        _, pendulum_angle, arm_rate, pendulum_rate = state
        torque = jnp.clip(action[0], -self.max_torque, self.max_torque)

        mass = self.pendulum_mass
        arm_length = self.arm_length
        com_length = self.pendulum_com
        coupled_inertia = self.pendulum_inertia
        cross_inertia = mass * arm_length * com_length
        arm_axis_inertia = self.arm_inertia
        sin_angle = jnp.sin(pendulum_angle)
        cos_angle = jnp.cos(pendulum_angle)

        mass_matrix = jnp.array(
            [
                [
                    arm_axis_inertia + coupled_inertia * sin_angle**2,
                    cross_inertia * cos_angle,
                ],
                [cross_inertia * cos_angle, coupled_inertia],
            ]
        )
        generalized_force = jnp.array(
            [
                torque
                - self.arm_damping * arm_rate
                - 2.0
                * coupled_inertia
                * sin_angle
                * cos_angle
                * pendulum_rate
                * arm_rate
                + cross_inertia * sin_angle * pendulum_rate**2,
                coupled_inertia * sin_angle * cos_angle * arm_rate**2
                + mass * 9.81 * com_length * sin_angle
                - self.pendulum_damping * pendulum_rate,
            ]
        )
        acceleration = jnp.linalg.solve(mass_matrix, generalized_force)
        return jnp.array(
            [arm_rate, pendulum_rate, acceleration[0], acceleration[1]]
        )

    def dynamics(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        k1 = self._derivative(state, action)
        k2 = self._derivative(state + 0.5 * self.dt * k1, action)
        k3 = self._derivative(state + 0.5 * self.dt * k2, action)
        k4 = self._derivative(state + self.dt * k3, action)
        next_state = state + self.dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        next_state = next_state.at[0].set(wrap_angle(next_state[0]))
        return next_state.at[1].set(wrap_angle(next_state[1]))

    def running_cost(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
        target_state: jnp.ndarray,
    ) -> jnp.ndarray:
        arm_error = wrap_angle(state[0] - target_state[0])
        pendulum_error = wrap_angle(state[1] - target_state[1])
        arm_limit = 5.0 * jnp.pi / 6.0
        constraint_violation = jnp.logical_or(
            arm_error < -arm_limit,
            arm_error > arm_limit,
        )
        return (
            50.0 * (jnp.cos(arm_error) - 1.0) ** 2
            + 500.0 * (jnp.cos(pendulum_error) - 1.0) ** 2
            + state[2] ** 2
            + state[3] ** 2
            + 50000.0 * constraint_violation.astype(state.dtype)
        )

    def terminal_cost(
        self,
        state: jnp.ndarray,
        final_target_state: jnp.ndarray,
    ) -> jnp.ndarray:
        return jnp.zeros((), dtype=state.dtype)


def save_plots(
    times: np.ndarray,
    states: np.ndarray,
    torques: np.ndarray,
    sample_counts: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True)
    axes[0].plot(times, states[:, 0], label="arm angle")
    axes[0].plot(times, states[:, 1], label="pendulum angle")
    axes[0].axhline(0.0, color="black", linewidth=1, linestyle="--")
    axes[0].set_ylabel("Angle [rad]")
    axes[0].legend()

    axes[1].plot(times, states[:, 2], label="arm rate")
    axes[1].plot(times, states[:, 3], label="pendulum rate")
    axes[1].set_ylabel("Rate [rad/s]")
    axes[1].legend()

    axes[2].step(times[:-1], torques, where="post", color="tab:green")
    axes[2].set_ylabel("Torque [N m]")

    for component, sigma in enumerate((REFERENCE_SIGMA,)):
        axes[3].step(
            times[:-1],
            sample_counts[:, component],
            where="post",
            label=f"sigma={sigma:g}",
        )
    axes[3].set_ylabel("Samples")
    axes[3].set_xlabel("Time [s]")
    axes[3].legend()

    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.suptitle("Log-MPPI Furuta pendulum swing-up")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_animation(
    states: np.ndarray,
    model: FurutaPendulumModel,
    output_path: Path,
) -> None:
    figure = plt.figure(figsize=(7, 7))
    axis = figure.add_subplot(111, projection="3d")
    radius = model.arm_length + model.pendulum_length + 0.02
    axis.set(
        xlim=(-radius, radius),
        ylim=(-radius, radius),
        zlim=(-model.pendulum_length - 0.02, model.pendulum_length + 0.02),
        xlabel="x [m]",
        ylabel="y [m]",
        zlabel="z [m]",
    )
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=25, azim=45)
    arm_line, = axis.plot([], [], [], "o-", linewidth=4, color="tab:blue")
    pendulum_line, = axis.plot([], [], [], "o-", linewidth=3, color="tab:red")

    def update(frame):
        arm_angle, pendulum_angle = states[frame, :2]
        arm_end = model.arm_length * np.array(
            [np.cos(arm_angle), np.sin(arm_angle), 0.0]
        )
        tangent = np.array([-np.sin(arm_angle), np.cos(arm_angle), 0.0])
        pendulum_end = arm_end + model.pendulum_length * (
            np.sin(pendulum_angle) * tangent
            + np.cos(pendulum_angle) * np.array([0.0, 0.0, 1.0])
        )
        arm_line.set_data([0.0, arm_end[0]], [0.0, arm_end[1]])
        arm_line.set_3d_properties([0.0, 0.0])
        pendulum_line.set_data(
            [arm_end[0], pendulum_end[0]],
            [arm_end[1], pendulum_end[1]],
        )
        pendulum_line.set_3d_properties([arm_end[2], pendulum_end[2]])
        axis.set_title(f"t={frame * DT:.2f} s")
        return arm_line, pendulum_line

    animation = FuncAnimation(
        figure,
        update,
        frames=len(states),
        interval=DT * 1000.0,
        blit=False,
    )
    animation.save(output_path, writer="ffmpeg", fps=round(1.0 / DT), dpi=120)
    plt.close(figure)


def main() -> None:
    result_dir = (
        Path(__file__).resolve().parents[2]
        / "result"
        / "furuta_pendulum"
        / "logmppi"
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    model = FurutaPendulumModel()
    solver = LogMPPISolver(
        model=model,
        horizon=HORIZON,
        num_samples=NUM_SAMPLES,
        temperature=TEMPERATURE,
        noise_sigma=NOISE_SIGMA,
        log_noise_mean=LOG_NOISE_MEAN,
        log_noise_std=LOG_NOISE_STD,
    )

    target_state = jnp.zeros(4)
    reference = jnp.tile(target_state, (HORIZON, 1))
    state = jnp.array([0.0, jnp.pi, 0.0, 0.0])
    mean_actions = jnp.zeros((HORIZON, 1))

    warm = solver.step(
        jax.random.PRNGKey(SEED + 1),
        state,
        mean_actions,
        reference,
        return_weights=True,
        return_cost=True,
    )
    jax.block_until_ready(warm)

    key = jax.random.PRNGKey(SEED)
    states = [np.asarray(state)]
    torques = []
    sample_counts = []
    ess_ratios = []
    predicted_costs = []
    print("=== Log-MPPI Furuta pendulum swing-up start ===")
    for step in range(SIM_STEPS):
        key, subkey = jax.random.split(key)
        action, mean_actions, weights, predicted_cost = solver.step(
            subkey,
            state,
            mean_actions,
            reference,
            return_weights=True,
            return_cost=True,
        )
        state = model.dynamics(state, action)
        jax.block_until_ready(state)
        states.append(np.asarray(state))
        torques.append(float(action[0]))
        sample_counts.append(np.array([NUM_SAMPLES]))
        ess_ratios.append(normalized_ess(np.asarray(weights)))
        predicted_costs.append(float(predicted_cost))
        if step % 25 == 24:
            print(
                f"{step + 1:3d}/{SIM_STEPS} | "
                f"arm={float(state[0]):+.3f} | "
                f"pendulum={float(state[1]):+.3f} | "
                f"torque={float(action[0]):+.4f} | "
                f"ESS/K={ess_ratios[-1]:.4f} | cost={predicted_costs[-1]:.1f}"
            )

    states = np.asarray(states)
    torques = np.asarray(torques)
    sample_counts = np.asarray(sample_counts)
    ess_ratios = np.asarray(ess_ratios)
    predicted_costs = np.asarray(predicted_costs)
    times = np.arange(len(states)) * DT
    save_plots(
        times,
        states,
        torques,
        sample_counts,
        result_dir / "logmppi_furuta_results.png",
    )
    save_diagnostics(
        result_dir,
        "Log-MPPI",
        times[:-1],
        ess_ratios,
        predicted_costs,
    )
    if SAVE_ANIMATION:
        save_animation(states, model, result_dir / "logmppi_furuta.mp4")
    print(f"[saved] {result_dir}")


if __name__ == "__main__":
    main()
