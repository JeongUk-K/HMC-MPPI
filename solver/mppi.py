import jax
import jax.numpy as jnp
from functools import partial
from collsion_checker.collision_model import CollisionModel
from model.model_base import ModelBase

class MPPISolver:
    def __init__(
        self, 
        model: ModelBase, 
        horizon: int, 
        num_samples: int, 
        temperature: float, 
        noise_sigma: float | tuple[float, ...],
        collision_checker: CollisionModel = None,
        collision_penalty: float = 1000.0,
    ):
        self.model = model
        self.horizon = horizon
        self.num_samples = num_samples
        self.temperature = temperature
        self.noise_sigma = noise_sigma
        self.collision_checker = collision_checker
        self.collision_penalty = collision_penalty

    @partial(jax.jit, static_argnums=(0, 5, 7))
    def step(self, key: jax.Array, current_state: jnp.ndarray, mean_action_seq: jnp.ndarray, ref_trajectory: jnp.ndarray, return_weights: bool = False, collision_context=None, return_cost: bool = False):
        """
        Args:
            ref_trajectory: (Horizon, State_dim) 형태의 참조 궤적.
        """
        action_dim = mean_action_seq.shape[-1]

        # 1. 무작위 노이즈 생성 및 샘플 궤적 구축
        noise_sigma = jnp.asarray(self.noise_sigma, dtype=mean_action_seq.dtype)
        noise = jax.random.normal(
            key,
            shape=(self.num_samples, self.horizon, action_dim),
        ) * noise_sigma
        
        # 2. 모델에 정의된 제한값을 가져와 제어 입력 클램핑 적용
        sampled_actions = jnp.clip(
            mean_action_seq + noise, 
            self.model.action_min, 
            self.model.action_max
        )

        # 3. 단일 궤적 롤아웃 함수
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
            
            # 제어 시퀀스와 참조 궤적을 동시에 순회
            final_state, costs = jax.lax.scan(scan_step, current_state, (actions_seq, ref_trajectory))
            terminal_cost = self.model.terminal_cost(final_state, ref_trajectory[-1])
            if self.collision_checker is not None:
                terminal_cost += self.collision_checker.collision_cost(
                    final_state,
                    self.collision_penalty,
                    collision_context,
                )
            total_cost = jnp.sum(costs) + terminal_cost
            return total_cost

        # 4. K개의 샘플 궤적 병렬 롤아웃 (vmap)
        trajectory_costs = jax.vmap(rollout_fn)(sampled_actions)

        # 5. 가중치 계산
        min_cost = jnp.min(trajectory_costs)
        weights = jax.nn.softmax(-(trajectory_costs - min_cost) / self.temperature)

        # 6. 가중 평균을 통한 새로운 제어열 계산
        optimal_action_seq = jnp.sum(weights[:, None, None] * sampled_actions, axis=0)
        
        # 7. Warm-start 준비: 마지막 최적 입력을 새 horizon 끝까지 유지
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
