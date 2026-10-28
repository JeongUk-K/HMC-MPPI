import jax
import jax.numpy as jnp
from functools import partial
from jax.scipy.special import logsumexp
from collsion_checker.collision_model import CollisionModel
from model.model_base import ModelBase

# 가우시안 혼합 proposal을 위한 샘플 수 적응 배분.

class GRMPPISolver:
    """잡음 성분별 샘플 수를 적응적으로 배분하는 GR-MPPI 솔버.

    target 분포와 proposal 분포는 다음 가우시안 혼합으로 표현된다.

        q_target(eps)   = sum_m alpha_m q_m(eps)
        q_proposal(eps) = sum_m rho_m q_m(eps)

    여기서 q_m은 표준편차 sigma_m을 갖는 평균 0의 가우시안 분포이다.
    전체 샘플 수는 고정하고, 제어 step 사이에서 각 성분의 샘플 수 K_m만
    변경한다.
    """

    def __init__(
        self,
        model: ModelBase,
        horizon: int,
        num_samples: int,
        temperature: float,
        noise_sigmas: tuple[float | tuple[float, ...], ...],
        target_weights: tuple[float, ...] = None,
        defensive_ratio: float = 0.2,
        min_samples_per_component: int = 1,
        collision_checker: CollisionModel = None,
        collision_penalty: float = 1000.0,
    ):
        """고정된 솔버 설정을 저장하고 target weight를 정규화한다.

        인자:
            model: 동역학, 실행 비용, 종단 비용, 제어 입력 제한을 정의한 모델.
            horizon: 한 예측 궤적에 포함되는 제어 입력의 개수.
            num_samples: 한 MPPI 갱신에서 사용할 전체 샘플 수 N.
            temperature: Boltzmann 온도 lambda.
            noise_sigmas: 각 성분의 scalar sigma_m 또는 action 차원별 vector sigma_m.
            target_weights: 고정된 target 혼합 weight alpha_m.
            defensive_ratio: alpha_m 탐색을 유지하기 위한 방어적 혼합 비율 eta.
            min_samples_per_component: 각 성분에 보장할 최소 샘플 수 K_m.

        target weight는 다음과 같이 정규화한다.

            alpha_m <- alpha_m / sum_j alpha_j.
        """
        self.model = model
        self.horizon = horizon
        self.num_samples = num_samples
        self.temperature = temperature
        self.noise_sigmas = tuple(noise_sigmas)
        self.num_components = len(noise_sigmas)
        self.defensive_ratio = defensive_ratio
        self.min_samples_per_component = min_samples_per_component
        self.collision_checker = collision_checker
        self.collision_penalty = collision_penalty

        if target_weights is None:
            target_weights = (1.0 / self.num_components,) * self.num_components
        target_weight_sum = sum(target_weights)
        self.target_weights = tuple(
            weight / target_weight_sum for weight in target_weights
        )

    def _allocate_samples(self, proposal_weights):
        """Proposal weight rho_m을 정수 샘플 수 K_m으로 변환한다.

        먼저 모든 성분에 K_min개의 샘플을 예약한다. 남은 샘플은 각 성분의
        초과 요구량에 비례하여 배분한다.

            K_m^desired = N rho_m
            e_m = max(K_m^desired - K_min, 0)
            K_m^extra ~= (N - M K_min) e_m / sum_j e_j.

        floor 연산 후 남는 샘플은 소수 부분이 큰 성분부터 하나씩 배정한다.
        따라서 일반적으로 다음 조건을 만족한다.

            sum_m K_m = N,    K_m >= K_min,
            rho_m^actual = K_m / N.
        """
        proposal_weights = proposal_weights / jnp.sum(proposal_weights)
        reserved_samples = (
            self.num_components * self.min_samples_per_component
        )
        free_samples = self.num_samples - reserved_samples

        if free_samples == 0:
            counts = jnp.full(
                self.num_components,
                self.min_samples_per_component,
                dtype=jnp.int32,
            )
            return counts, counts / self.num_samples

        desired_counts = proposal_weights * self.num_samples
        residual_weights = jnp.maximum(
            desired_counts - self.min_samples_per_component,
            0.0,
        )
        real_extra_counts = (
            residual_weights
            / jnp.sum(residual_weights)
            * free_samples
        )
        extra_counts = jnp.floor(real_extra_counts).astype(jnp.int32)

        remainder = free_samples - jnp.sum(extra_counts)
        fractional = real_extra_counts - extra_counts
        order = jnp.argsort(-fractional)
        extras = jnp.zeros(self.num_components, dtype=jnp.int32)
        extras = extras.at[order].set(
            (jnp.arange(self.num_components) < remainder).astype(jnp.int32)
        )

        counts = (
            extra_counts + extras + self.min_samples_per_component
        )
        actual_proposal_weights = counts / self.num_samples
        return counts, actual_proposal_weights

    def _sample_noise(self, key, action_dim, dtype, proposal_weights):
        """적응적으로 배분된 가우시안 성분에서 잡음(noise)을 생성한다.

        샘플 i의 성분 인덱스를 c_i라고 하면 다음과 같이 noise를 생성한다.

            z_i ~ Normal(0, I),
            eps_i = diag(sigma_{c_i}) z_i.

        scalar sigma이면 모든 action 차원에 같은 값을 쓰고, vector sigma이면
        action 차원마다 서로 다른 표준편차를 갖는 대각 공분산을 사용한다.

        샘플은 성분별로 연속해서 저장한다. `component_ids`에는 c_i를 저장하여
        롤아웃 이후 성분별 통계를 계산할 수 있도록 한다.
        `actual_proposal_weights`는 K_m/N이며, 현재 실제 생성된 샘플에 대응하는
        proposal 혼합 weight이다.
        """
        counts, actual_proposal_weights = self._allocate_samples(
            proposal_weights
        )

        boundaries = jnp.cumsum(counts)[:-1]
        sample_indices = jnp.arange(self.num_samples)[:, None]
        component_ids = jnp.sum(
            sample_indices >= boundaries[None, :],
            axis=1,
        )

        sigmas = self._sigma_matrix(action_dim, dtype)
        standard_noise = jax.random.normal(
            key,
            shape=(self.num_samples, self.horizon, action_dim),
            dtype=dtype,
        )
        noise = standard_noise * sigmas[component_ids, None, :]

        return noise, component_ids, counts, actual_proposal_weights

    def _sigma_matrix(self, action_dim, dtype):
        """Scalar sigma는 등방성, vector sigma는 대각 공분산으로 해석한다."""
        sigmas = jnp.asarray(self.noise_sigmas, dtype=dtype)
        if sigmas.ndim == 1:
            sigmas = jnp.broadcast_to(
                sigmas[:, None],
                (self.num_components, action_dim),
            )
        return sigmas

    def _log_density_ratio(self, noise, proposal_weights):
        """모든 잡음 궤적에 대한 log target/proposal 확률 밀도 비율을 계산한다.

        action 차원 a의 분산을 sigma_{m,a}^2라고 하면 각 성분의 로그
        확률밀도는 다음과 같다.

            log q_m(eps)
              = -1/2 sum_{t,a} [eps_{t,a}^2 / sigma_{m,a}^2
                                 + log(2 pi sigma_{m,a}^2)].

        중요도 표본추출(importance sampling) 보정에 사용하는 비율은 다음과 같다.

            q_target(eps) / q_proposal(eps)
              = [sum_m alpha_m q_m(eps)]
                / [sum_m rho_m q_m(eps)].

        수치적 안정성을 위해 `logsumexp`를 사용하여 로그 공간에서 계산한다.
        """
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

        target_weights = jnp.asarray(
            self.target_weights,
            dtype=noise.dtype,
        )
        log_target_density = logsumexp(
            jnp.log(target_weights)[None, :] + log_component_density,
            axis=1,
        )
        log_proposal_density = logsumexp(
            jnp.log(proposal_weights)[None, :] + log_component_density,
            axis=1,
        )
        return log_target_density - log_proposal_density

    def _update_proposal_weights(
        self,
        trajectory_costs,
        component_ids,
        counts,
    ):
        """각 성분의 선호도(desirability)로부터 다음 혼합 weight를 추정한다.

        비용이 S_i인 궤적 i에 대해 Boltzmann 선호도를 계산한다.

            d_i = exp(-(S_i - S_min) / lambda).

        성분 m의 표본 평균은 해당 성분의 분배 함수값(partition value)을 추정한다.

            beta_m ~= (1 / K_m) sum_{i:c_i=m} d_i.

        이후 posterior 성분 weight는 다음과 같다.

            rho_m^post = alpha_m beta_m / sum_j alpha_j beta_j.

        마지막으로 방어적 혼합(defensive mixing)을 적용하여 특정 성분으로의
        붕괴를 막는다.

            rho_{m,next} = (1 - eta) rho_m^post + eta alpha_m.

        S_min을 빼는 것은 모든 d_i에 동일한 양의 배율을 곱하는 것과 같아서
        정규화 시 상쇄된다. 동시에 수치적 언더플로(underflow)를 줄일 수 있다.
        """
        shifted_desirability = jnp.exp(
            -(trajectory_costs - jnp.min(trajectory_costs))
            / self.temperature
        )
        component_sums = jnp.zeros(
            self.num_components,
            dtype=trajectory_costs.dtype,
        )
        component_sums = component_sums.at[component_ids].add(
            shifted_desirability
        )
        component_partition = component_sums / counts

        target_weights = jnp.asarray(
            self.target_weights,
            dtype=trajectory_costs.dtype,
        )
        posterior_component_weights = target_weights * component_partition
        posterior_component_weights /= jnp.sum(posterior_component_weights)

        return (
            (1.0 - self.defensive_ratio) * posterior_component_weights
            + self.defensive_ratio * target_weights
        )

    @partial(jax.jit, static_argnums=(0, 6, 8))
    def step(
        self,
        key,
        current_state,
        mean_action_seq,
        ref_trajectory,
        proposal_weights,
        return_weights=False,
        collision_context=None,
        return_cost=False,
    ):
        """이동 지평선(receding horizon) 방식으로 GR-MPPI를 한 번 갱신한다.

        갱신 과정은 다음과 같다.

        1. K_m개의 샘플을 성분별로 배분하고 혼합 잡음 eps_i를 생성한다.
        2. 제한된 후보 제어 입력 u_i = clip(u_bar + eps_i)를 만든다.
        3. 모든 후보를 롤아웃하여 궤적 비용 S_i를 계산한다.
        4. 다음 Boltzmann-importance weight를 계산한다.

               w_i proportional to exp(-S_i / lambda)
                         * q_target(eps_i) / q_proposal(eps_i).

        5. 다음 식으로 nominal control sequence를 갱신한다.

               u_bar_new = sum_i softmax(log w_i) u_i.

        6. 다음 반복에서 사용할 proposal weight rho_{next}를 추정한다.
        7. 첫 번째 action을 반환하고 나머지 제어열을 이동하여 다음
           이동 지평선 갱신의 웜 스타트로 사용한다.

        반환값:
            action: 현재 시스템에 적용할 첫 번째 제어 입력.
            next_mean_action_seq: 한 칸 이동된 기준 제어열.
            next_proposal_weights: 다음 호출에 사용할 rho_{next}.
            counts: 현재 반복에서 성분별로 사용한 정수 샘플 수 [K_1, ..., K_M].
        """
        action_dim = mean_action_seq.shape[-1]

        # 1. rho_t에 따라 각 sigma의 샘플 수 K_m을 배분하고 잡음을 생성.
        noise, component_ids, counts, actual_proposal_weights = (
            self._sample_noise(
                key,
                action_dim,
                mean_action_seq.dtype,
                proposal_weights,
            )
        )
        # 후보 제어 입력: u_i = clip(u_bar + eps_i).
        sampled_actions = jnp.clip(
            mean_action_seq + noise,
            self.model.action_min,
            self.model.action_max,
        )

        # 2. 모든 샘플 궤적을 병렬 롤아웃하여 궤적 비용 S_i를 계산.
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

        # 3. Boltzmann 비용과 target/proposal 혼합 분포의 비율을 결합.
        #    log(w_i) = -S_i / lambda
        #              + log(q_target(eps_i) / q_proposal(eps_i)).
        log_weights = (
            -trajectory_costs / self.temperature
            + self._log_density_ratio(noise, actual_proposal_weights)
        )
        weights = jax.nn.softmax(log_weights)

        # 4. 정규화된 weight로 기준 제어열을 갱신.
        #    u_bar_new = sum_i softmax(log(w_i)) * u_i.
        optimal_action_seq = jnp.sum(
            weights[:, None, None] * sampled_actions,
            axis=0,
        )

        # 5. beta_m을 추정하여 다음 proposal weight rho_{t+1}을 계산.
        next_proposal_weights = self._update_proposal_weights(
            trajectory_costs,
            component_ids,
            counts,
        )

        # 6. 첫 제어 입력을 반환하고 나머지 제어열은 한 칸 당겨 웜 스타트.
        action = optimal_action_seq[0]
        next_mean_action_seq = jnp.concatenate(
            (optimal_action_seq[1:], optimal_action_seq[-1:]),
            axis=0,
        )

        outputs = (
            action,
            next_mean_action_seq,
            next_proposal_weights,
            counts,
        )
        if return_weights:
            outputs += (weights,)
        if return_cost:
            outputs += (jnp.min(trajectory_costs),)
        return outputs
