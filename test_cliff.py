import pytest
import jax, jax.numpy as jnp
from cliff_jax import Cliff

def test_construct_cliff():
  Cliff.full()

def test_step_to_finish():
  env = Cliff.full()
  state = Cliff.InternalState(state=jnp.asarray(15), step=jnp.asarray(0))
  state, reward, done = env.step(state, 1)
  assert state == Cliff.InternalState(state=jnp.asarray(23), step=jnp.asarray(0))
  assert reward == 50
  assert done

if __name__ == "__main__":
  pytest.main([__file__])
