import json
from pathlib import Path

import numpy as np


STATE_WEIGHTS = np.array([5.0, 10.0, 0.1, 0.1])
SETTLING_LIMITS = np.array([0.5, 0.1, 0.5, 0.5])
SETTLING_HOLD_SECONDS = 0.5


def _wrapped_errors(states: np.ndarray) -> np.ndarray:
    errors = states.copy()
    errors[:, 1] = (errors[:, 1] + np.pi) % (2.0 * np.pi) - np.pi
    return errors


def _find_settling_index(
    errors: np.ndarray,
    delta_t: float,
    start_index: int = 0,
) -> int | None:
    stable = np.all(np.abs(errors) <= SETTLING_LIMITS, axis=1)
    hold_steps = max(1, int(round(SETTLING_HOLD_SECONDS / delta_t)))
    for candidate in range(start_index, len(stable) - hold_steps + 1):
        if np.all(stable[candidate : candidate + hold_steps]):
            return candidate
    return None


def save_evaluation(
    result_dir: Path,
    method: str,
    display_name: str,
    time_history,
    state_history,
    force_times,
    force_history,
    cost_times,
    cost_history,
    delta_t: float,
    fault_time: float,
    impulse_theta_dot: float,
) -> None:
    """Save one impulse-recovery run in the cartpole_eval artifact format."""
    times = np.asarray(time_history, dtype=float)
    states = np.asarray(state_history, dtype=float)
    force_times = np.asarray(force_times, dtype=float)
    forces = np.asarray(force_history, dtype=float)
    cost_times = np.asarray(cost_times, dtype=float)
    costs = np.asarray(cost_history, dtype=float)
    errors = _wrapped_errors(states)
    weighted_squared_error = np.sum(STATE_WEIGHTS * errors**2, axis=1)
    weighted_error = np.sqrt(weighted_squared_error)

    fault_step = int(round(fault_time / delta_t))
    pre_fault_cost_index = max(0, min(fault_step - 1, len(costs) - 1))
    fault_cost_index = min(fault_step, len(costs) - 1)
    response_state_index = min(fault_step + 1, len(states) - 1)
    initial_index = _find_settling_index(
        errors[: fault_step + 1], delta_t
    )
    recovery_index = _find_settling_index(
        errors, delta_t, response_state_index
    )
    recovery_end = recovery_index + 1 if recovery_index is not None else len(states)
    final_steps = min(len(states), max(1, int(round(2.0 / delta_t))))

    metrics = {
        "method": method,
        "display_name": display_name,
        "success": initial_index is not None and recovery_index is not None,
        "initial_swingup_success": initial_index is not None,
        "initial_settling_time_s": (
            float(times[initial_index]) if initial_index is not None else None
        ),
        "recovery_success": recovery_index is not None,
        "recovery_time_s": (
            float(times[recovery_index] - fault_time)
            if recovery_index is not None
            else None
        ),
        "disturbance_start_time_s": fault_time,
        "disturbance_type": "theta_dot_impulse",
        "disturbance_delta_theta_dot_rad_s": impulse_theta_dot,
        "recovery_weighted_iae": float(
            np.sum(weighted_error[response_state_index:recovery_end]) * delta_t
        ),
        "disturbance_response_total_weighted_iae": float(
            np.sum(weighted_error[response_state_index:]) * delta_t
        ),
        "peak_disturbance_response_weighted_error": float(
            np.max(weighted_error[response_state_index:])
        ),
        "total_weighted_ise": float(np.sum(weighted_squared_error) * delta_t),
        "final_2s_weighted_rmse": float(
            np.sqrt(np.mean(weighted_error[-final_steps:] ** 2))
        ),
        "max_abs_cart_position_m": float(np.max(np.abs(states[:, 0]))),
        "max_abs_pole_rate_rad_s": float(np.max(np.abs(states[:, 3]))),
        "control_rms_n": float(np.sqrt(np.mean(forces**2))),
        "control_energy_n2_s": float(np.sum(forces**2) * delta_t),
        "disturbance_response_control_energy_n2_s": float(
            np.sum(forces[fault_step:] ** 2) * delta_t
        ),
        "control_max_abs_n": float(np.max(np.abs(forces))),
        "control_total_variation_n": float(np.sum(np.abs(np.diff(forces)))),
        "solver_best_cost_initial": float(costs[0]),
        "solver_best_cost_pre_impulse": float(costs[pre_fault_cost_index]),
        "solver_best_cost_at_impulse": float(costs[fault_cost_index]),
        "solver_best_cost_final": float(costs[-1]),
        "solver_best_cost_min": float(np.min(costs)),
        "settling_definition": {
            "limits": dict(
                zip(("x", "theta", "x_dot", "theta_dot"), SETTLING_LIMITS.tolist())
            ),
            "hold_time_s": SETTLING_HOLD_SECONDS,
        },
        "configuration": {
            "dt_s": delta_t,
            "duration_s": float(times[-1]),
        },
    }

    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        result_dir / "trajectory.npz",
        time_s=times,
        states=states,
        weighted_error=weighted_error,
        force_time_s=force_times,
        force_n=forces,
        solver_cost_time_s=cost_times,
        solver_best_cost=costs,
    )
    print(f"[INFO] Evaluation data saved to {result_dir}")
