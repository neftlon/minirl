import jax, jax.numpy as jnp, functools
from .util import Buf, get_episode_reward, reduce_episodes

@functools.partial(jax.jit, static_argnames=["model"])
def expected_reward(model, params, buf_state: Buf.State) -> jax.Array:
  # compute sum of logps for each episode
  def accumulate_logps(buf_state: Buf.State, ep_idx, offset, logps):
    obs, act = buf_state.observations[offset], buf_state.actions[offset]
    logp = model.logp(params, obs, act) # compute \pi(s_t\vert a_t)
    return logps.at[ep_idx].set(logps[ep_idx] + logp)
    
  weights = get_episode_reward(buf_state)
  logps = reduce_episodes(accumulate_logps, jnp.zeros_like(buf_state.ep_ends, dtype=float), buf_state)
  # weight logps and compute average to approximate E[J(\pi)] over all episodes
  return jnp.sum(weights * logps) / buf_state.num_eps
