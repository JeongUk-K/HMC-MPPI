import jax
import jax.numpy as jnp
from functools import partial
from collsion_checker.collision_model import CollisionModel
from model.model_base import ModelBase


class LogMPPISolver:
    def __init__(
        self,
        model: ModelBase,
        horizon: int,
        num_samples: int,
        temperature: float,
        noise_sigma: float | tuple[float, ...],
        log_noise_mean: float,
        log_noise_std: float,
        collision_checker: CollisionModel = None,
        collision_penalty: float = 1000.0,
    ):
        self.model = model
        self.horizon = horizon
        self.num_samples = num_samples
        self.temperature = temperature
        self.noise_sigma = noise_sigma
        self.log_noise_mean = log_noise_mean
        self.log_noise_std = log_noise_std
        self.collision_checker = collision_checker
        self.collision_penalty = collision_penalty

    @partial(jax.jit, static_argnums=(0, 5, 7))
    def step(
        self,
        key,
        current_state,
        mean_action_seq,
        ref_trajectory,
        return_weights=False,
        collision_context=None,
        return_cost=False,
    ):
        action_dim = mean_action_seq.shape[-1]

        # 1. Normal-Lognormal(NLN) 제어 노이즈 샘플링
        normal_key, lognormal_key = jax.random.split(key)
        normal_noise = jax.random.normal(
            normal_key,
            shape=(self.num_samples, self.horizon, action_dim),
            dtype=mean_action_seq.dtype,
        )
        lognormal_noise = jnp.exp(
            self.log_noise_mean
            + self.log_noise_std
            * jax.random.normal(
                lognormal_key,
                shape=(self.num_samples, self.horizon, action_dim),
                dtype=mean_action_seq.dtype,
            )
        )
        noise_sigma = jnp.asarray(self.noise_sigma, dtype=mean_action_seq.dtype)
        noise = noise_sigma * normal_noise * lognormal_noise
        sampled_actions = jnp.clip(
            mean_action_seq + noise,
            self.model.action_min,
            self.model.action_max,
        )

        # 2. 샘플 궤적 롤아웃
        def rollout_fn(actions_seq):
            def scan_step(state, inputs):
                action, target_state = inputs
                next_state = self.model.dynamics(state, action)
                cost = self.model.running_cost(state, action, target_state)
                if self.collision_checker is not None:
                    cost += self.collision_checker.collision_cost(
                        state,
                        self.collision_penalty,
                        collision_context,
                    )
                return next_state, cost

            final_state, costs = jax.lax.scan(
                scan_step,
                current_state,
                (actions_seq, ref_trajectory),
            )
            terminal_cost = self.model.terminal_cost(
                final_state,
                ref_trajectory[-1],
            )
            if self.collision_checker is not None:
                terminal_cost += self.collision_checker.collision_cost(
                    final_state,
                    self.collision_penalty,
                    collision_context,
                )
            return jnp.sum(costs) + terminal_cost

        trajectory_costs = jax.vmap(rollout_fn)(sampled_actions)

        # 3. Boltzmann weight 계산
        min_cost = jnp.min(trajectory_costs)
        weights = jax.nn.softmax(
            -(trajectory_costs - min_cost) / self.temperature
        )

        # 4. 가중 평균으로 제어열 갱신
        optimal_action_seq = jnp.sum(
            weights[:, None, None] * sampled_actions,
            axis=0,
        )

        # 5. 첫 제어 입력을 반환하고 마지막 최적 입력을 확장해 warm-start
        action = optimal_action_seq[0]
        next_mean_action_seq = jnp.concatenate(
            (optimal_action_seq[1:], optimal_action_seq[-1:]),
            axis=0,
        )

        outputs = (action, next_mean_action_seq)
        if return_weights:
            outputs += (weights,)
        if return_cost:
            outputs += (min_cost,)
        return outputs
