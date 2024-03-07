import jax, jax.numpy as jnp, jax.random as jr
import typing, dataclasses, functools

@dataclasses.dataclass(frozen=True)
class Env:
  class InternalState: ...

  class StepResult(typing.NamedTuple):
    state: typing.Any # InternalState
    reward: jax.Array # float
    done: jax.Array # bool

  # convert internal state to observable state
  def observe(self, internal_state: "Env.InternalState") -> typing.Any: ...

  def step(self, internal_state: "Env.InternalState", action) -> "Env.StepResult": ...

  def reset(self, key: jax.Array) -> "Env.InternalState": ...
