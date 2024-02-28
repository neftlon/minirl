import pytest
import jax.numpy as jnp, jax.random as jr
from jax.experimental import io_callback
from util import Buf, reduce_episodes

@pytest.mark.parametrize("obs_shape", [(), (2,), (3,4)])
def test_obs_shape(obs_shape):
  key = jr.key(0)
  keys = jr.split(key, 3)

  max_num_eps, max_episode_len = 3, 4
  num_steps = 2
  obs = jr.randint(keys[0], (num_steps,) + obs_shape, 1, 10)
  action = jr.randint(keys[1], (num_steps,), 0, 5)
  reward = jr.uniform(keys[2], (num_steps,), minval=-5, maxval=5)

  buf = Buf(max_num_eps, max_episode_len, obs_shape=obs_shape)
  buf_state = buf.empty()
  for t in zip(obs, action, reward):
    buf_state = buf.append(buf_state, *t)
  buf_state = buf.end_episode(buf_state)

  assert buf_state.num_eps == 1
  assert buf_state.offset == 2

  # the first two element should get assigned to the observations
  assert jnp.allclose(buf_state.observations[:num_steps], obs)

  # check by iteration
  def f(buf_state: Buf.State, epidx, offset, carry):
    if epidx < 1:
      # there is only one episode inside
      assert buf_state.observations[offset].shape == obs_shape
    return carry
  # NB: wrap with io_callback to allow arbitray python control flow in checker function
  assert reduce_episodes(lambda *args: io_callback(f, (), *args), (), buf_state) == ()

if __name__ == "__main__":
  pytest.main([__file__])