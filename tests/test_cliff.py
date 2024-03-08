import pytest
import jax.numpy as jnp
from minirl.cliff import Cliff

def test_step_to_finish():
  env = Cliff()
  state = Cliff.InternalState(state=jnp.asarray(35), step=jnp.asarray(0))
  state, reward, done = env.step(state, 1)
  assert state == Cliff.InternalState(state=jnp.asarray(47), step=jnp.asarray(0))
  assert reward == (env.reward_scheme.win_reward + env.reward_scheme.step_penalty)
  assert done

if __name__ == "__main__":
  pytest.main([__file__])
