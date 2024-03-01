import pytest
import jax.numpy as jnp, jax.random as jr
from jax.experimental import io_callback
from minirl.util import Buf, reduce_episodes

@pytest.mark.parametrize("obs_shape", [(), (2,), (3,4)])
def test_obs_shape(obs_shape):
  key = jr.key(0)
  keys = jr.split(key, 3)

  buf_size, max_episode_len = 12, 3
  num_steps = 2
  obs = jr.randint(keys[0], (num_steps,) + obs_shape, 1, 10)
  action = jr.randint(keys[1], (num_steps,), 0, 5)
  reward = jr.uniform(keys[2], (num_steps,), minval=-5, maxval=5)

  buf = Buf(buf_size, max_episode_len, obs_shape=obs_shape)
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
  assert reduce_episodes(lambda *args: io_callback(f, (), *args), (), buf, buf_state) == ()

@pytest.mark.parametrize("buf_state,expected_rtg", [
  (
    Buf.State(
      offset=jnp.asarray(10),
      num_eps=jnp.asarray(3),
      ep_ends=jnp.array([2, 6, 10, 0]),
      # NB: observations and actions are not valid, but this does not matter for reward calculation
      observations=jnp.zeros(()),
      actions=jnp.zeros(()),
      rewards=jnp.array([-1, 3, 2, -1, -2, -1, -1, 3, 2, -1, 0], dtype=float),
    ),
    jnp.array([2, 3, -2, -4, -3, -1, 3, 4, 1, -1, 0], dtype=float),
  ),
  (
    Buf.State(
      offset=jnp.asarray(4),
      num_eps=jnp.asarray(2),
      ep_ends=jnp.asarray([2,4]),
      observations=jnp.zeros(()),
      actions=jnp.zeros(()),
      rewards=jnp.asarray([-1,2,-1,-1,0,0], dtype=float),
    ),
    jnp.asarray([1,2,-2,-1,0,0], dtype=float)
  ),
  (
    Buf.State(
      offset=jnp.asarray(18),
      num_eps=jnp.asarray(2),
      ep_ends=jnp.asarray([9,18]),
      observations=jnp.zeros(()),
      actions=jnp.zeros(()),
      rewards=jnp.array([-1,-1,-1,-1,-1,-1,-1,-1,50,-1,-1,-1,-1,-1,-1,-1,-1,50,0,0], dtype=float)
    ),
    jnp.array([42,43,44,45,46,47,48,49,50,42,43,44,45,46,47,48,49,50,0,0], dtype=float)
  ),
])
def test_compute_rtg(buf_state, expected_rtg):
  rtg = buf_state.get_reward_to_go()
  print(rtg)
  print(expected_rtg)
  assert jnp.allclose(rtg, expected_rtg)

def test_fill():
  buf = Buf(4, 2)
  buf_state = buf.empty()
  
  # append first episode
  buf_state = buf.append(buf_state, 1, 1, 1.)
  buf_state = buf.append(buf_state, 2, 2, 2.)
  buf_state = buf.end_episode(buf_state)
  assert buf.can_append_episode(buf_state)

  # append second episode
  buf_state = buf.append(buf_state, 3, 3, 3.)
  buf_state = buf.append(buf_state, 4, 4, 4.)
  buf_state = buf.end_episode(buf_state)
  assert not buf.can_append_episode(buf_state)

  # check contents
  assert jnp.allclose(buf_state.observations, jnp.arange(1,5))
  assert jnp.allclose(buf_state.actions, jnp.arange(1,5))
  assert jnp.allclose(buf_state.rewards, jnp.arange(1.,5.))

if __name__ == "__main__":
  pytest.main([__file__])