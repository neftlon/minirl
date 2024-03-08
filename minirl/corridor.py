import jax.numpy as jnp, jax.random as jr
from .util import Env

class Corridor:
  """
  Short corridor grid with switched actions from "Reinforcement Learning: An Introduction by Sutton and Barto"

  Actions: L/0 or R/1
  States: 0,1,2,3 (last one is terminal)
  """
  def observe(self, state):
    return state # state is already an integer

  def step(self, state, action) -> Env.StepResult:
    d = 2 * action - 1 # 0 -> -1, 1 -> 1
    d = jnp.where(state == 1, -d, d) # invert action in first state
    newstate = state + d # propose move
    newstate = jnp.maximum(jnp.minimum(newstate, 3), 0) # enforce bounds
    return Env.StepResult(
      state=newstate,
      reward=jnp.array(-1, dtype=int),
      done=newstate == 3,
    )
  
  def reset(self, _key):
    return jnp.array(0, dtype=int)
  