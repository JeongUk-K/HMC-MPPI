import jax
import jax.numpy as jnp
from functools import partial
from jax.scipy.special import logsumexp
from collsion_checker.collision_model import CollisionModel
from model.model_base import ModelBase


class GRMPPISolver:
    def __init__(
        self,
        model: ModelBase,
        horizon: int,
        num_samples: int,
        temperature: float,
        noise_sigmas: tuple[float | tuple[float, ...], ...],
        sample_allocation: tuple[int, ...] = None,
        target_weights: tuple[float, ...] = None,
        collision_checker: CollisionModel = None,
        collision_penalty: float = 1000.0,
    ):
        self.model = model
        self.horizon = horizon
        self.num_samples = num_samples
        self.temperature = temperature
        self.noise_sigmas = tuple(noise_sigmas)
        self.num_components = len(noise_sigmas)
        self.collision_checker = collision_checker
        self.collision_penalty = collision_penalty

        if sample_allocation is None:
            base, remainder = divmod(num_samples, self.num_components)
            sample_allocation = tuple(
                base + int(i < remainder) for i in range(self.num_components)
            )
        self.sample_allocation = tuple(sample_allocation)

        if target_weights is None:
            target_weights = (1.0 / self.num_components,) * self.num_components
        target_weight_sum = sum(target_weights)
        self.target_weights = tuple(
            weight / target_weight_sum for weight in target_weights
        )
        self.proposal_weights = tuple(
            count / num_samples for count in self.sample_allocation
        )

    def _sample_noise(self, key, action_dim, dtype):
        keys = jax.random.split(key, self.num_components)
        sigmas = self._sigma_matrix(action_dim, dtype)
        noise = [
            jax.random.normal(
                keys[i],
                shape=(self.sample_allocation[i], self.horizon, action_dim),
                dtype=dtype,
            )
            * sigmas[i]
            for i in range(self.num_components)
        ]
        return jnp.concatenate(noise, axis=0)

    def _sigma_matrix(self, action_dim, dtype):
        """Scalar sigma는 등방성, vector sigma는 대각 공분산으로 해석한다."""
        sigmas = jnp.asarray(self.noise_sigmas, dtype=dtype)
        if sigmas.ndim == 1:
            sigmas = jnp.broadcast_to(
                sigmas[:, None],
                (self.num_components, action_dim),
            )
        return sigmas

    def _log_density_ratio(self, noise):
        sigmas = self._sigma_matrix(noise.shape[-1], noise.dtype)
        variances = sigmas**2
        scaled_squared_norm = jnp.sum(
            noise[:, None, :, :] ** 2 / variances[None, :, None, :],
            axis=(2, 3),
        )
        log_normalizer = self.horizon * jnp.sum(
            jnp.log(2.0 * jnp.pi * variances),
            axis=1,
        )
        log_component_density = -0.5 * (
            scaled_squared_norm + log_normalizer[None, :]
        )

        log_target_density = logsumexp(
            jnp.log(jnp.asarray(self.target_weights))[None, :]
            + log_component_density,
            axis=1,
        )
        log_proposal_density = logsumexp(
            jnp.log(jnp.asarray(self.proposal_weights))[None, :]
            + log_component_density,
            axis=1,
        )
        return log_target_density - log_proposal_density

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

        # 1. 각 sigma에서 정해진 개수만큼 노이즈 샘플링
        noise = self._sample_noise(key, action_dim, mean_action_seq.dtype)
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

        # 3. Boltzmann 비용과 target/proposal mixture 비율 결합
        log_weights = (
            -trajectory_costs / self.temperature
            + self._log_density_ratio(noise)
        )
        weights = jax.nn.softmax(log_weights)

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
            outputs += (jnp.min(trajectory_costs),)
        return outputs
