import abc
import jax.numpy as jnp

class ModelBase(abc.ABC):
    """
    모든 제어 대상 시스템이 상속받아야 하는 추상 클래스 템플릿입니다.
    """
    
    @property
    @abc.abstractmethod
    def action_min(self) -> jnp.ndarray:
        """제어 입력의 최솟값을 반환합니다."""
        pass

    @property
    @abc.abstractmethod
    def action_max(self) -> jnp.ndarray:
        """제어 입력의 최댓값을 반환합니다."""
        pass

    @abc.abstractmethod
    def dynamics(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        """
        현재 상태와 제어 입력을 받아 다음 상태를 반환합니다.
        x_{t+1} = f(x_t, u_t)
        """
        pass

    @abc.abstractmethod
    def running_cost(self, state: jnp.ndarray, action: jnp.ndarray, target_state: jnp.ndarray) -> jnp.ndarray:
        """
        매 스텝 발생하는 비용(Cost)을 계산합니다.
        - target_state: 해당 타임스텝에서 추종해야 할 목표 상태값
        """
        pass

    def terminal_cost(self, state: jnp.ndarray, final_target_state: jnp.ndarray) -> jnp.ndarray:
        """
        예측 구간의 마지막 스텝에서 발생하는 비용입니다.
        필요에 따라 오버라이딩하여 사용합니다. 기본값은 0.0입니다.
        """
        return jnp.array(0.0)