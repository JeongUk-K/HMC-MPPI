import abc
from collections.abc import Sequence

import jax.numpy as jnp

from model.model_base import ModelBase


class CollisionModel(abc.ABC):
    """Model state의 일부 성분만 사용하여 충돌을 판정하는 기본 클래스.

    예를 들어 state=[x, y, yaw, velocity]에서 위치만 사용하려면
    ``state_indices=(0, 1)``로 설정한다. 단일 state ``(state_dim,)``와
    배치 state ``(..., state_dim)``를 모두 지원한다.
    """

    def __init__(
        self,
        model: ModelBase,
        state_indices: Sequence[int],
    ) -> None:
        if not state_indices:
            raise ValueError("state_indices must contain at least one index")
        if not all(isinstance(index, int) for index in state_indices):
            raise TypeError("state_indices must contain only integers")

        self.model = model
        self.state_indices = tuple(state_indices)

    def select_collision_state(self, state: jnp.ndarray) -> jnp.ndarray:
        """전체 state에서 충돌 판정에 사용할 성분만 선택한다."""
        if state.ndim == 0:
            raise ValueError("state must have at least one dimension")

        state_dim = state.shape[-1]
        if any(
            index < -state_dim or index >= state_dim
            for index in self.state_indices
        ):
            raise IndexError(
                f"state index is out of bounds for state dimension {state_dim}"
            )

        return jnp.take(
            state,
            jnp.asarray(self.state_indices),
            axis=-1,
        )

    def is_collision(
        self,
        state: jnp.ndarray,
        collision_context=None,
    ) -> jnp.ndarray:
        """전체 state와 가변 map/context를 받아 충돌 여부를 반환한다."""
        collision_state = self.select_collision_state(state)
        return self.check_collision(collision_state, collision_context)

    def collision_cost(
        self,
        state: jnp.ndarray,
        penalty: float,
        collision_context=None,
    ) -> jnp.ndarray:
        """충돌 여부를 MPPI running cost에 더할 penalty로 변환한다."""
        return (
            self.is_collision(state, collision_context).astype(state.dtype)
            * penalty
        )

    @abc.abstractmethod
    def check_collision(
        self,
        collision_state: jnp.ndarray,
        collision_context,
    ) -> jnp.ndarray:
        """선택된 state에 대한 실제 충돌 판정을 하위 클래스에서 구현한다.

        collision_context에는 ROS에서 갱신되는 occupancy map, map origin,
        resolution 등의 JAX PyTree를 전달할 수 있다.
        반환 shape은 입력의 batch shape와 같아야 한다. 단일 state를 받으면
        scalar bool, 배치 state를 받으면 batch별 bool 배열을 반환한다.
        """
        raise NotImplementedError
