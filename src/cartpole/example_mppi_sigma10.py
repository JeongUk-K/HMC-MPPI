import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    from cartpole import CartPole
else:
    from .cartpole import CartPole
from model.model_base import ModelBase
from solver.mppi import MPPISolver


class CartPoleModel(ModelBase):
    def __init__(
        self,
        delta_t: float = 0.02,
        mass_of_cart: float = 1.0,
        mass_of_pole: float = 0.01,
        length_of_pole: float = 2.0,
        max_force_abs: float = 100.0,
    ) -> None:
        self.delta_t = delta_t
        self.mass_of_cart = mass_of_cart
        self.mass_of_pole = mass_of_pole
        self.length_of_pole = length_of_pole
        self.max_force_abs = max_force_abs
        self.stage_cost_weight = jnp.array([5.0, 10.0, 0.1, 0.1])
        self.terminal_cost_weight = jnp.array([5.0, 10.0, 0.1, 0.1])

    @property
    def action_min(self) -> jnp.ndarray:
        return jnp.array([-self.max_force_abs])

    @property
    def action_max(self) -> jnp.ndarray:
        return jnp.array([self.max_force_abs])

    def dynamics(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x, theta, x_dot, theta_dot = state
        force = jnp.clip(action[0], -self.max_force_abs, self.max_force_abs)

        g = 9.81
        cart_mass = self.mass_of_cart
        pole_mass = self.mass_of_pole
        pole_length = self.length_of_pole

        temp = (
            force
            - pole_mass * pole_length * theta_dot**2 * jnp.sin(theta)
        ) / (cart_mass + pole_mass)
        theta_ddot = (
            g * jnp.sin(theta) + jnp.cos(theta) * temp
        ) / (
            pole_length
            * (
                4.0 / 3.0
                - pole_mass * jnp.cos(theta) ** 2 / (cart_mass + pole_mass)
            )
        )
        x_ddot = temp + pole_mass * pole_length * theta_ddot * jnp.cos(theta) / (
            cart_mass + pole_mass
        )

        next_x = x + x_dot * self.delta_t
        next_theta = theta + theta_dot * self.delta_t
        next_theta = (next_theta + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
        next_x_dot = x_dot + x_ddot * self.delta_t
        next_theta_dot = theta_dot + theta_ddot * self.delta_t

        return jnp.array([next_x, next_theta, next_x_dot, next_theta_dot])

    def running_cost(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
        target_state: jnp.ndarray,
    ) -> jnp.ndarray:
        state_error = state - target_state
        theta_error = (state_error[1] + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
        state_error = state_error.at[1].set(theta_error)
        return jnp.sum(self.stage_cost_weight * state_error**2)

    def terminal_cost(
        self,
        state: jnp.ndarray,
        final_target_state: jnp.ndarray,
    ) -> jnp.ndarray:
        state_error = state - final_target_state
        theta_error = (state_error[1] + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
        state_error = state_error.at[1].set(theta_error)
        return jnp.sum(self.terminal_cost_weight * state_error**2)


def plot_results(
    time_history: np.ndarray,
    state_history: np.ndarray,
    error_history: np.ndarray,
    force_times: np.ndarray,
    force_history: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    axes[0].plot(time_history, state_history[:, 0], label="x [m]")
    axes[0].plot(time_history, state_history[:, 1], label="theta [rad]")
    axes[0].plot(time_history, state_history[:, 2], label="x_dot [m/s]")
    axes[0].plot(
        time_history,
        state_history[:, 3],
        label="theta_dot [rad/s]",
    )
    axes[0].set_title("CartPole State")
    axes[0].set_ylabel("State")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(time_history, error_history, color="tab:red")
    axes[1].set_title("State Error")
    axes[1].set_ylabel("||state - target||")
    axes[1].grid(True)

    axes[2].step(force_times, force_history, where="post", color="tab:green")
    axes[2].set_title("Control Force")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Force [N]")
    axes[2].grid(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.show()
    plt.close(fig)
    print(f"[INFO] Plot saved to {output_path}")


def main() -> None:
    delta_t = 0.02
    horizon_len = 80
    sim_steps = 400
    result_dir = Path(__file__).resolve().parents[2] / "result" / "cartpole" / "mppi_sigma_10"
    result_dir.mkdir(parents=True, exist_ok=True)

    model = CartPoleModel(
        delta_t=delta_t,
        mass_of_cart=1.0,
        mass_of_pole=0.01,
        length_of_pole=2.0,
        max_force_abs=100.0,
    )
    cartpole = CartPole(
        mass_of_cart=1.0,
        mass_of_pole=0.01,
        length_of_pole=2.0,
        max_force_abs=100.0,
        delta_t=delta_t,
        visualize=True,
    )
    cartpole.reset(np.array([0.0, np.pi, 0.0, 0.0]))

    solver = MPPISolver(
        model=model,
        horizon=horizon_len,
        num_samples=1000,
        temperature=50.0,
        noise_sigma=10.0,
    )

    key = jax.random.PRNGKey(42)
    target_state = jnp.zeros(4)
    ref_trajectory = jnp.tile(target_state, (horizon_len, 1))
    mean_action_seq = jnp.zeros((horizon_len, 1))
    initial_state = jnp.asarray(cartpole.get_state())
    time_history = [0.0]
    state_history = [cartpole.get_state()]
    error_history = [float(jnp.linalg.norm(initial_state - target_state))]
    force_times = []
    force_history = []

    print("=== CartPole MPPI Simulation Start ===")

    for step in range(sim_steps):
        key, subkey = jax.random.split(key)
        state = jnp.asarray(cartpole.get_state())
        action, mean_action_seq = solver.step(
            subkey,
            state,
            mean_action_seq,
            ref_trajectory,
        )
        cartpole.update(u=[float(action[0])], delta_t=delta_t)
        state = jnp.asarray(cartpole.get_state())
        error = jnp.linalg.norm(state - target_state)
        time_history.append((step + 1) * delta_t)
        state_history.append(cartpole.get_state())
        error_history.append(float(error))
        force_times.append((step + 1) * delta_t)
        force_history.append(float(action[0]))

        print(
            f"Step {step:03d} | "
            f"State(x,theta,x_dot,theta_dot): "
            f"({state[0]:+.3f}, {state[1]:+.3f}, "
            f"{state[2]:+.3f}, {state[3]:+.3f}) | "
            f"Force: {action[0]:+.3f} | Error: {error:.3f}"
        )

    output_path = result_dir / "mppi_cartpole.mp4"
    cartpole.save_animation(
        str(output_path),
        interval=int(delta_t * 1000),
        movie_writer="ffmpeg",
    )
    print(f"[INFO] Animation saved to {output_path}")

    plot_results(
        np.asarray(time_history),
        np.asarray(state_history),
        np.asarray(error_history),
        np.asarray(force_times),
        np.asarray(force_history),
        result_dir / "mppi_cartpole_results.png",
    )


if __name__ == "__main__":
    main()
