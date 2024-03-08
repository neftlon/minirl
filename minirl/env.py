import jax, jax.numpy as jnp, typing
from .util import Env

class RestrictTime(typing.NamedTuple):
  "Environment wrapper to restrict time of RL environments."
  env: typing.Any
  max_steps: int

  class State(typing.NamedTuple):
    steps: jax.Array
    env_state: typing.Any

  def observe(self, state):
    return self.env.observe(state.env_state)
  
  def step(self, internal_state: "RestrictTime.State", action: typing.Any) -> Env.StepResult:
    env_result = self.env.step(internal_state.env_state, action)
    outoftime = internal_state.steps >= self.max_steps
    done = env_result.done | outoftime # out-of-time causes the environment to be done
    new_state = RestrictTime.State(
      # increase steps only if not done
      steps=jnp.where(done, internal_state.steps, internal_state.steps + 1),
      env_state=env_result.state,
    )
    return Env.StepResult(state=new_state, reward=env_result.reward, done=done)

  def reset(self, key) -> "RestrictTime.State":
    env_state = self.env.reset(key)
    return RestrictTime.State(jnp.array(0, dtype=int), env_state)

