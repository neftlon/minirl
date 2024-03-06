import jax, jax.numpy as jnp, functools
from .util import Buf, get_episode_reward, reduce_episodes

@functools.partial(jax.jit, static_argnames=["buf", "model"])
def ep_weighted_expected_reward(model, model_params, model_state, buf: Buf, buf_state: Buf.State) -> jax.Array:
  r"Approximate $J(\pi)$ using episode-weighted ($R(\tau)$) sum of log-ps."
  # compute sum of logps for each episode
  def accumulate_logps(buf_state: Buf.State, ep_idx, offset, logps):
    obs, act = buf_state.observations[offset], buf_state.actions[offset]
    logp = model.logp(model_params, model_state, obs, act) # compute \pi(s_t\vert a_t)
    return logps.at[ep_idx].set(logps[ep_idx] + logp)
    
  weights = get_episode_reward(buf, buf_state) # one weight per episode
  logps = reduce_episodes(accumulate_logps, jnp.zeros_like(buf_state.ep_ends, dtype=float), buf, buf_state)
  # weight logps and compute average to approximate E[J(\pi)] over all episodes
  return jnp.sum(weights * logps) / buf_state.num_eps

def weighted_expected_reward(model, model_params, model_state, buf: Buf, buf_state: Buf.State, weights: jax.Array) -> jax.Array:
  r"Approximate $J(\pi)$ using individually weighed sum of log-ps."
  mask = jnp.arange(buf.buf_size) < buf_state.offset
  logps = jax.vmap(model.logp, (None,None,0,0))(model_params, model_state, buf_state.observations, buf_state.actions)
  weighted_logps = jnp.where(mask, weights * logps, 0)
  # weight logps and compute average to approximate E[J(\pi)] over all episodes
  return jnp.sum(weighted_logps) / buf_state.num_eps

@functools.partial(jax.jit, static_argnames=["buf", "model"])
def rtg_expected_reward(model, model_params, model_state, buf: Buf, buf_state: Buf.State) -> jax.Array:
  r"Approximate $J(\pi)$ using log-ps weighted by reward-to-go."
  weights = buf_state.get_reward_to_go() # one weight for each logp
  return weighted_expected_reward(model, model_params, model_state, buf, buf_state, weights)
