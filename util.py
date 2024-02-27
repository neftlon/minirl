import jax, jax.numpy as jnp
import typing

class Buf(typing.NamedTuple):
  max_num_eps: jax.Array # int
  max_episode_len: jax.Array # int

  class State(typing.NamedTuple):
    offset: jax.Array # int, current location in the buffer
    num_eps: jax.Array # int, number of episodes that is contained in this buffer
    ep_ends: jax.Array # end markers for episodes in the buffer (exclusive)
    observations: typing.Any
    actions: typing.Any
    rewards: typing.Any

    @property
    def ep_starts(self) -> jax.Array:
      starts = jnp.zeros_like(self.ep_ends)
      return starts.at[1:].set(self.ep_ends[:-1])
    
    @property
    def ep_lens(self) -> jax.Array:
      return self.ep_ends - self.ep_starts
    
    @property
    def buf_size(self) -> int:
      return len(self.observations)

    def reset(self) -> "Buf.State":
      return Buf.State(
        offset=jnp.asarray(0),
        num_eps=jnp.asarray(0),
        ep_ends=self.ep_ends,
        observations=self.observations,
        actions=self.actions,
        rewards=self.rewards,
      )
  
  def can_append_episode(self, state: "State") -> bool:
    free = state.buf_size - state.offset
    return (state.num_eps < self.max_num_eps) & (free >= self.max_episode_len)

  def append(self, state: "State", obs, action, reward) -> "State":
    newoffset = state.offset + 1
    in_bounds = newoffset < len(state.observations)
    return Buf.State(
      offset=jnp.where(in_bounds, newoffset, state.offset),
      num_eps=state.num_eps,
      ep_ends=state.ep_ends,
      observations=jnp.where(in_bounds, state.observations.at[state.offset].set(obs), state.observations),
      actions=jnp.where(in_bounds, state.actions.at[state.offset].set(action), state.actions),
      rewards=jnp.where(in_bounds, state.rewards.at[state.offset].set(reward), state.rewards),
    )

  def end_episode(self, state: "State") -> "State":
    # Safety: calling can_append_episode before appending/ending an episode ensures that there is enough space
    return Buf.State(
      offset=state.offset,
      num_eps=state.num_eps + 1,
      ep_ends=state.ep_ends.at[state.num_eps].set(state.offset),
      observations=state.observations,
      actions=state.actions,
      rewards=state.rewards,
    )
  
  def empty(self, buf_size: typing.Optional[int] = None) -> "State":
    if buf_size is None:
      buf_size = self.max_num_eps * self.max_episode_len
    return Buf.State(
      offset=jnp.asarray(0),
      num_eps=jnp.asarray(0),
      ep_ends=jnp.zeros(self.max_num_eps, dtype=int),
      observations=jnp.zeros(buf_size, dtype=int),
      actions=jnp.zeros(buf_size, dtype=int),
      rewards=jnp.zeros(buf_size),
    )

def reduce_episodes(fn, carry_init, buf_state: Buf.State):
  class LoopState(typing.NamedTuple):
    carry: typing.Any
    epidx: jax.Array # int

  def body(offset: int, state: LoopState):
    not_done = offset < buf_state.offset # buf's current level is not reached yet
    next_offset_overflow = (offset + 1) >= buf_state.ep_ends[state.epidx] # increase eps idx preventive
    return LoopState(
      carry=jnp.where(not_done, fn(buf_state, state.epidx, offset, state.carry), state.carry),
      epidx=jnp.where(not_done & next_offset_overflow, state.epidx + 1, state.epidx)
    )
  
  state = LoopState(carry_init, jnp.asarray(0))
  state = jax.lax.fori_loop(0, buf_state.buf_size, body, state)
  return state.carry

# compute sum of rewards for each episode
def accumulate_rewards(buf_state, ep_idx, offset, ep_rew):
  reward = buf_state.rewards[offset]
  return ep_rew.at[ep_idx].set(ep_rew[ep_idx] + reward)

def get_episode_reward(buf_state: Buf.State):
  "Returns cumulative reward for each episode"
  return reduce_episodes(accumulate_rewards, jnp.zeros_like(buf_state.ep_ends, dtype=float), buf_state)
