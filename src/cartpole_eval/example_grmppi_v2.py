import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from model.model_base import ModelBase
from solver.gr_mppi_v2 import GRMPPISolver


METHOD = "gr_mppi_v2"
DISPLAY_NAME = "GR-MPPI adaptive K"
ADAPTIVE_SAMPLING = True
DT = 0.02
HORIZON = 100
NUM_SAMPLES = 1000
TEMPERATURE = 50.0
SIM_STEPS = 500
SEED = 42
STATE_WEIGHTS = np.array([5.0, 10.0, 0.1, 0.1])
STATE_NAMES = ("x", "theta", "x_dot", "theta_dot")
SETTLING_LIMITS = np.array([0.5, 0.1, 0.5, 0.5])
SETTLING_HOLD_SECONDS = 0.5
DISTURBANCE_START_TIME_S = 3.0
DISTURBANCE_DURATION_S = 1.0
DISTURBANCE_END_TIME_S = (
    DISTURBANCE_START_TIME_S + DISTURBANCE_DURATION_S
)
DISTURBANCE_START_STEP = int(round(DISTURBANCE_START_TIME_S / DT))
DISTURBANCE_END_STEP = int(round(DISTURBANCE_END_TIME_S / DT))
DISTURBANCE_THETA_DDOT = 6.0


class CartPoleModel(ModelBase):
    """이 평가 파일 안에서 완결되는 CartPole 동역학/비용 모델."""

    def __init__(
        self,
        delta_t=DT,
        mass_of_cart=1.0,
        mass_of_pole=0.01,
        length_of_pole=2.0,
        max_force_abs=100.0,
    ):
        self.delta_t = delta_t
        self.mass_of_cart = mass_of_cart
        self.mass_of_pole = mass_of_pole
        self.length_of_pole = length_of_pole
        self.max_force_abs = max_force_abs
        self.stage_cost_weight = jnp.asarray(STATE_WEIGHTS)
        self.terminal_cost_weight = jnp.asarray(STATE_WEIGHTS)

    @property
    def action_min(self):
        return jnp.array([-self.max_force_abs])

    @property
    def action_max(self):
        return jnp.array([self.max_force_abs])

    def dynamics(self, state, action):
        x, theta, x_dot, theta_dot = state
        force = jnp.clip(action[0], -self.max_force_abs, self.max_force_abs)
        cart_mass = self.mass_of_cart
        pole_mass = self.mass_of_pole
        pole_length = self.length_of_pole

        temp = (
            force
            - pole_mass * pole_length * theta_dot**2 * jnp.sin(theta)
        ) / (cart_mass + pole_mass)
        theta_ddot = (
            9.81 * jnp.sin(theta) + jnp.cos(theta) * temp
        ) / (
            pole_length
            * (
                4.0 / 3.0
                - pole_mass * jnp.cos(theta) ** 2
                / (cart_mass + pole_mass)
            )
        )
        x_ddot = (
            temp
            + pole_mass * pole_length * theta_ddot * jnp.cos(theta)
            / (cart_mass + pole_mass)
        )

        next_state = jnp.array(
            [
                x + x_dot * self.delta_t,
                theta + theta_dot * self.delta_t,
                x_dot + x_ddot * self.delta_t,
                theta_dot + theta_ddot * self.delta_t,
            ]
        )
        wrapped_theta = (
            (next_state[1] + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
        )
        return next_state.at[1].set(wrapped_theta)

    @staticmethod
    def _state_error(state, target_state):
        error = state - target_state
        wrapped_theta = (error[1] + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
        return error.at[1].set(wrapped_theta)

    def running_cost(self, state, action, target_state):
        del action
        error = self._state_error(state, target_state)
        return jnp.sum(self.stage_cost_weight * error**2)

    def terminal_cost(self, state, final_target_state):
        error = self._state_error(state, final_target_state)
        return jnp.sum(self.terminal_cost_weight * error**2)


def make_solver(model):
    return GRMPPISolver(
        model=model,
        horizon=HORIZON,
        num_samples=NUM_SAMPLES,
        temperature=TEMPERATURE,
        noise_sigmas=(5.0, 10.0, 15.0),
        target_weights=(1.0, 1.0, 1.0),
        defensive_ratio=0.2,
        min_samples_per_component=100,
    )


def wrapped_errors(states):
    errors = states.copy()
    errors[:, 1] = (errors[:, 1] + np.pi) % (2.0 * np.pi) - np.pi
    return errors


def find_settling_index(errors, start_index=0):
    stable = np.all(np.abs(errors) <= SETTLING_LIMITS, axis=1)
    hold_steps = int(round(SETTLING_HOLD_SECONDS / DT))
    for candidate in range(
        start_index,
        len(stable) - hold_steps + 1,
    ):
        if np.all(stable[candidate : candidate + hold_steps]):
            return candidate
    return None


def scalar_stats(values):
    if len(values) == 0:
        return {"mean": None, "std": None, "rmse": None, "p95_abs": None}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "rmse": float(np.sqrt(np.mean(values**2))),
        "p95_abs": float(np.percentile(np.abs(values), 95)),
    }


def calculate_metrics(states, forces, solver_times_ms, ess_history):
    errors = wrapped_errors(states)
    weighted_squared_error = np.sum(STATE_WEIGHTS * errors**2, axis=1)
    weighted_error = np.sqrt(weighted_squared_error)
    initial_settling_index = find_settling_index(
        errors[: DISTURBANCE_START_STEP + 1]
    )
    recovery_index = find_settling_index(errors, DISTURBANCE_END_STEP)
    initial_success = initial_settling_index is not None
    recovery_success = recovery_index is not None
    success = initial_success and recovery_success
    transient_end = (
        initial_settling_index + 1
        if initial_success
        else DISTURBANCE_START_STEP + 1
    )
    recovery_end = recovery_index + 1 if recovery_success else len(states)
    steady_errors = (
        errors[recovery_index:] if recovery_success else np.empty((0, 4))
    )
    steady_weighted = (
        weighted_error[recovery_index:]
        if recovery_success
        else np.empty(0)
    )
    final_window_steps = int(round(2.0 / DT))
    final_errors = errors[-final_window_steps:]
    final_weighted = weighted_error[-final_window_steps:]

    steady_state = {
        name: scalar_stats(steady_errors[:, index])
        for index, name in enumerate(STATE_NAMES)
    }
    final_state = {
        name: scalar_stats(final_errors[:, index])
        for index, name in enumerate(STATE_NAMES)
    }

    return {
        "method": METHOD,
        "display_name": DISPLAY_NAME,
        "success": bool(success),
        "settling_time_s": (
            float(initial_settling_index * DT)
            if initial_success
            else None
        ),
        "initial_swingup_success": bool(initial_success),
        "initial_settling_time_s": (
            float(initial_settling_index * DT)
            if initial_success
            else None
        ),
        "recovery_success": bool(recovery_success),
        "recovery_time_s": (
            float((recovery_index - DISTURBANCE_END_STEP) * DT)
            if recovery_success
            else None
        ),
        "recovery_settling_absolute_time_s": (
            float(recovery_index * DT) if recovery_success else None
        ),
        "disturbance_start_time_s": DISTURBANCE_START_TIME_S,
        "disturbance_duration_s": DISTURBANCE_DURATION_S,
        "disturbance_end_time_s": DISTURBANCE_END_TIME_S,
        "disturbance_theta_ddot_rad_s2": DISTURBANCE_THETA_DDOT,
        "settling_definition": {
            "limits": dict(zip(STATE_NAMES, SETTLING_LIMITS.tolist())),
            "hold_time_s": SETTLING_HOLD_SECONDS,
        },
        "transient_weighted_iae": float(
            np.sum(weighted_error[:transient_end]) * DT
        ),
        "recovery_weighted_iae": float(
            np.sum(
                weighted_error[DISTURBANCE_START_STEP:recovery_end]
            )
            * DT
        ),
        "disturbance_response_total_weighted_iae": float(
            np.sum(weighted_error[DISTURBANCE_START_STEP:]) * DT
        ),
        "peak_disturbance_response_weighted_error": float(
            np.max(weighted_error[DISTURBANCE_START_STEP:])
        ),
        "total_weighted_ise": float(np.sum(weighted_squared_error) * DT),
        "steady_weighted_rmse": (
            float(np.sqrt(np.mean(steady_weighted**2)))
            if recovery_success
            else None
        ),
        "steady_weighted_mean": (
            float(np.mean(steady_weighted)) if recovery_success else None
        ),
        "steady_weighted_p95": (
            float(np.percentile(steady_weighted, 95))
            if recovery_success
            else None
        ),
        "steady_state_error": steady_state,
        "final_2s_weighted_rmse": float(
            np.sqrt(np.mean(final_weighted**2))
        ),
        "final_2s_state_error": final_state,
        "max_abs_cart_position_m": float(np.max(np.abs(states[:, 0]))),
        "max_abs_pole_rate_rad_s": float(
            np.max(np.abs(states[:, 3]))
        ),
        "control_rms_n": float(np.sqrt(np.mean(forces**2))),
        "control_energy_n2_s": float(np.sum(forces**2) * DT),
        "disturbance_response_control_energy_n2_s": float(
            np.sum(forces[DISTURBANCE_START_STEP:] ** 2) * DT
        ),
        "control_max_abs_n": float(np.max(np.abs(forces))),
        "control_total_variation_n": float(np.sum(np.abs(np.diff(forces)))),
        "solver_time_mean_ms": float(np.mean(solver_times_ms)),
        "solver_time_std_ms": float(np.std(solver_times_ms)),
        "solver_time_p95_ms": float(
            np.percentile(solver_times_ms, 95)
        ),
        "solver_time_max_ms": float(np.max(solver_times_ms)),
        "solver_frequency_hz": float(1000.0 / np.mean(solver_times_ms)),
        "ess_mean": float(np.mean(ess_history)),
        "ess_min": float(np.min(ess_history)),
        "ess_p05": float(np.percentile(ess_history, 5)),
        "ess_ratio_mean": float(np.mean(ess_history) / NUM_SAMPLES),
        "configuration": {
            "dt_s": DT,
            "duration_s": SIM_STEPS * DT,
            "horizon": HORIZON,
            "num_samples": NUM_SAMPLES,
            "temperature": TEMPERATURE,
            "seed": SEED,
            "disturbance_start_time_s": DISTURBANCE_START_TIME_S,
            "disturbance_duration_s": DISTURBANCE_DURATION_S,
            "disturbance_theta_ddot_rad_s2": DISTURBANCE_THETA_DDOT,
            "noise_sigmas": [5.0, 10.0, 15.0],
            "defensive_ratio": 0.2,
            "min_samples_per_component": 100,
        },
        "timing_note": (
            "JAX compilation is warmed up before measurement. "
            "ESS calculation, host conversion, plant dynamics, metrics, "
            "plotting, and file I/O are outside solver timing."
        ),
    }


def save_plot(
    times,
    states,
    weighted_error,
    forces,
    ess_history,
    initial_settling_time,
    recovery_settling_time,
    sample_counts,
    output_path,
):
    rows = 5 if ADAPTIVE_SAMPLING else 4
    fig, axes = plt.subplots(rows, 1, figsize=(12, 3.2 * rows), sharex=True)

    for index, name in enumerate(STATE_NAMES):
        axes[0].plot(times, states[:, index], label=name)
    axes[0].set_title(f"{DISPLAY_NAME}: CartPole states")
    axes[0].set_ylabel("state")
    axes[0].grid(True)
    axes[0].legend(ncol=4)

    axes[1].plot(times, weighted_error, color="tab:red")
    axes[1].set_ylabel("weighted error")
    axes[1].grid(True)

    action_times = times[1:]
    axes[2].step(action_times, forces, where="post", color="tab:green")
    axes[2].set_ylabel("force [N]")
    axes[2].grid(True)

    axes[3].plot(action_times, ess_history, color="tab:purple")
    axes[3].axhline(
        NUM_SAMPLES,
        color="black",
        linestyle=":",
        linewidth=1,
        label="K",
    )
    axes[3].set_ylabel("ESS")
    axes[3].grid(True)
    axes[3].legend()

    if ADAPTIVE_SAMPLING:
        for component in range(sample_counts.shape[1]):
            axes[4].step(
                action_times,
                sample_counts[:, component],
                where="post",
                label=f"K{component + 1}",
            )
        axes[4].set_ylabel("sample K")
        axes[4].grid(True)
        axes[4].legend()

    for axis in axes:
        axis.axvline(
            DISTURBANCE_START_TIME_S,
            color="tab:orange",
            linestyle="-",
            linewidth=1,
        )
        axis.axvline(
            DISTURBANCE_END_TIME_S,
            color="tab:orange",
            linestyle="--",
            linewidth=1,
        )
        if initial_settling_time is not None:
            axis.axvline(
                initial_settling_time,
                color="tab:blue",
                linestyle="--",
                linewidth=1,
            )
        if recovery_settling_time is not None:
            axis.axvline(
                recovery_settling_time,
                color="tab:green",
                linestyle="--",
                linewidth=1,
            )
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    result_dir = (
        Path(__file__).resolve().parents[2]
        / "result"
        / "cartpole_eval"
        / METHOD
    )
    result_dir.mkdir(parents=True, exist_ok=True)

    model = CartPoleModel()
    solver = make_solver(model)
    target_state = jnp.zeros(4)
    ref_trajectory = jnp.tile(target_state, (HORIZON, 1))
    mean_action_seq = jnp.zeros((HORIZON, 1))
    state = jnp.array([0.0, jnp.pi, 0.0, 0.0])
    proposal_weights = jnp.ones(3) / 3.0
    key = jax.random.PRNGKey(SEED)

    # 동일한 shape의 진단 반환 경로를 먼저 컴파일하여 JIT 시간을 제외한다.
    warm_key = jax.random.PRNGKey(SEED + 10000)
    warm_outputs = solver.step(
        warm_key,
        state,
        mean_action_seq,
        ref_trajectory,
        proposal_weights,
        True,
    )
    jax.block_until_ready(warm_outputs)

    states = [np.asarray(state)]
    forces = []
    solver_times_ms = []
    ess_history = []
    sample_count_history = []

    print(f"=== {DISPLAY_NAME} evaluation start ===")
    for step in range(SIM_STEPS):
        if step == DISTURBANCE_START_STEP:
            print(
                f"[DISTURBANCE] t={DISTURBANCE_START_TIME_S:.2f}"
                f"~{DISTURBANCE_END_TIME_S:.2f}s, "
                f"theta_ddot={DISTURBANCE_THETA_DDOT:+.1f} rad/s^2"
            )
        key, subkey = jax.random.split(key)
        start = time.perf_counter()
        outputs = solver.step(
            subkey,
            state,
            mean_action_seq,
            ref_trajectory,
            proposal_weights,
            True,
        )
        jax.block_until_ready(outputs)
        solver_times_ms.append((time.perf_counter() - start) * 1000.0)
        (
            action,
            mean_action_seq,
            proposal_weights,
            sample_counts,
            weights,
        ) = outputs
        sample_count_history.append(np.asarray(sample_counts))

        # ESS 계산과 NumPy 변환은 의도적으로 solver 시간 측정 밖에 둔다.
        weights_np = np.asarray(weights)
        ess_history.append(float(1.0 / np.sum(weights_np**2)))
        forces.append(float(np.asarray(action)[0]))

        state = model.dynamics(state, action)
        if DISTURBANCE_START_STEP <= step < DISTURBANCE_END_STEP:
            state = state.at[3].add(DISTURBANCE_THETA_DDOT * DT)
        jax.block_until_ready(state)
        states.append(np.asarray(state))
        if step % 50 == 49:
            print(
                f"{step + 1:3d}/{SIM_STEPS} | "
                f"t={(step + 1) * DT:4.1f}s | "
                f"theta={float(state[1]):+.3f} | "
                f"solver={solver_times_ms[-1]:.3f} ms"
            )

    states = np.asarray(states)
    forces = np.asarray(forces)
    solver_times_ms = np.asarray(solver_times_ms)
    ess_history = np.asarray(ess_history)
    sample_counts = (
        np.asarray(sample_count_history)
        if sample_count_history
        else np.empty((SIM_STEPS, 0), dtype=int)
    )
    times = np.arange(SIM_STEPS + 1) * DT
    errors = wrapped_errors(states)
    weighted_error = np.sqrt(np.sum(STATE_WEIGHTS * errors**2, axis=1))
    metrics = calculate_metrics(
        states,
        forces,
        solver_times_ms,
        ess_history,
    )

    np.savez(
        result_dir / "trajectory.npz",
        time_s=times,
        states=states,
        wrapped_errors=errors,
        weighted_error=weighted_error,
        force_n=forces,
        solver_time_ms=solver_times_ms,
        ess=ess_history,
        sample_counts=sample_counts,
    )
    with (result_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    save_plot(
        times,
        states,
        weighted_error,
        forces,
        ess_history,
        metrics["initial_settling_time_s"],
        metrics["recovery_settling_absolute_time_s"],
        sample_counts,
        result_dir / "evaluation.png",
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"[saved] {result_dir}")


if __name__ == "__main__":
    main()
