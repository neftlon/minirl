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

class Buf(typing.NamedTuple):
  buf_size: int # size of buffer in steps
  max_episode_len: int
  obs_shape: tuple[int, ...] = () # shape of the observations

  class State(typing.NamedTuple):
    offset: jax.Array # int, current location in the buffer
    num_eps: jax.Array # int, number of episodes that is contained in this buffer
    ep_ends: jax.Array # end markers for episodes in the buffer (exclusive)
    observations: jax.Array
    actions: jax.Array
    rewards: jax.Array

    @property
    def ep_starts(self) -> jax.Array:
      starts = jnp.zeros_like(self.ep_ends)
      return starts.at[1:].set(self.ep_ends[:-1])
    
    @property
    def ep_lens(self) -> jax.Array:
      return self.ep_ends - self.ep_starts
    
    def get_avg_return(self):
      mask = jnp.arange(len(self.rewards)) < self.offset
      return jnp.sum(self.rewards[mask]) / self.num_eps

    
    def get_reward_to_go(self):
      class ScanState(typing.NamedTuple):
        epidx: jax.Array # int
        offset: jax.Array # int
        accum: jax.Array # float

      def f(state: ScanState, r: jax.Array):
        pass_border = (state.offset + 1) == self.ep_ends[state.epidx]
        accum = jnp.where(pass_border, 0., state.accum)
        epidx = jnp.where(pass_border, state.epidx - 1, state.epidx)
        accum += r
        return ScanState(epidx=epidx, offset=state.offset - 1, accum=accum), accum

      state = ScanState(
        epidx=self.num_eps - 1, # start at last episode
        offset=jnp.asarray(len(self.rewards) - 1),
        accum=jnp.asarray(0.),
      )
      _, rtg = jax.lax.scan(f, state, self.rewards, reverse=True)
      return rtg

    def reset(self) -> "Buf.State":
      return Buf.State(
        offset=jnp.asarray(0),
        num_eps=jnp.asarray(0),
        ep_ends=self.ep_ends,
        observations=self.observations,
        actions=self.actions,
        rewards=self.rewards,
      )

  def can_append_episode(self, state: "State") -> jax.Array: # bool
    free = self.buf_size - state.offset
    return free >= self.max_episode_len

  def append(self, state: "State", obs, action, reward) -> "State":
    newoffset = state.offset + 1
    # NB: newoffset can be one past the last writable index since below only the penultimate
    # index will be written to
    in_bounds = newoffset <= len(state.observations)
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
  
  def empty(self) -> "State":
    return Buf.State(
      offset=jnp.zeros((), dtype=int),
      num_eps=jnp.zeros((), dtype=int),
      # NB: in this buffer setting, it would be possible that each episode only occupies one slot
      ep_ends=jnp.zeros(self.buf_size, dtype=int),
      observations=jnp.zeros((self.buf_size,) + self.obs_shape, dtype=int),
      actions=jnp.zeros(self.buf_size, dtype=int),
      rewards=jnp.zeros(self.buf_size),
    )

def reduce_episodes(fn, carry_init, buf: Buf, buf_state: Buf.State):
  """
  Args:
    fn: (buf_state, epidx, offset, carry) -> newcarry

  Note: fn can be called with episode indices and offsets that are invalid and out of the buffer's range.
  """
  class LoopState(typing.NamedTuple):
    carry: typing.Any
    epidx: jax.Array # int

  def body(offset: int, state: LoopState):
    not_done = offset < buf_state.offset # buf's current level is not reached yet
    next_offset_overflow = (offset + 1) >= buf_state.ep_ends[state.epidx] # increase eps idx preventive
    carry = jax.tree_util.tree_map(
      lambda a, b: jnp.where(not_done, a, b),
      fn(buf_state, state.epidx, offset, state.carry),
      state.carry
    )
    epidx = jnp.where(not_done & next_offset_overflow, state.epidx + 1, state.epidx)
    return LoopState(carry=carry, epidx=epidx)
  
  state = LoopState(carry_init, jnp.asarray(0))
  state = jax.lax.fori_loop(0, buf.buf_size, body, state)
  return state.carry

# compute sum of rewards for each episode
def accumulate_rewards(buf_state, ep_idx, offset, ep_rew):
  reward = buf_state.rewards[offset]
  return ep_rew.at[ep_idx].set(ep_rew[ep_idx] + reward)

def get_episode_reward(buf: Buf, buf_state: Buf.State):
  "Returns cumulative reward for each episode"
  return reduce_episodes(accumulate_rewards, jnp.zeros_like(buf_state.ep_ends, dtype=float), buf, buf_state)

def run_episode(
  key, model, model_params, model_state, buf: Buf, buf_state: Buf.State, env: Env, max_episode_len: typing.Optional[int] = None,
) -> Buf.State:
  "Run one episode of `env` and store the results in `buf`/`buf_state`."
  class LoopState(typing.NamedTuple):
    key: jax.Array
    env_state: Env.InternalState
    buf_state: Buf.State
    done: jax.Array # bool
    timer: jax.Array # int

  def cond(state: LoopState):
    return ~state.done & ~(state.timer <= 0)
  
  def body(state: LoopState):
    obs = env.observe(state.env_state)
    key, action_key = jr.split(state.key)
    action = model(model_params, model_state, action_key, obs)
    env_state, reward, done = env.step(state.env_state, action)
    buf_state = buf.append(state.buf_state, obs, action, reward)
    timer = state.timer - (1 if max_episode_len is not None else 0)
    return LoopState(
      key=key,
      env_state=env_state,
      buf_state=buf_state,
      done=done,
      timer=timer,
    )

  key, env_key = jr.split(key)
  state = LoopState(
    key=key,
    env_state=env.reset(env_key),
    buf_state=buf_state,
    done=jnp.asarray(False),
    # set timer to 1 such that it never reaches 0 if environment is capable of tracking that itself
    timer=jnp.asarray(max_episode_len if isinstance(max_episode_len, int) else 1)
  )
  state = jax.lax.while_loop(cond, body, state)
  buf_state = buf.end_episode(state.buf_state) # finalize episode
  return buf_state

@functools.partial(jax.jit, static_argnames=["model", "env", "max_episode_len"])
def fill_buffer(
  key, model, model_params, model_state, buf: Buf, buf_state: Buf.State, env: Env, max_episode_len: typing.Optional[int] = None,
) -> Buf.State:
  "Run episodes until `buf`/`buf_state` cannot store another episode."
  class LoopState(typing.NamedTuple):
    key: jax.Array
    buf_state: Buf.State

  def cond(state: LoopState):
    return buf.can_append_episode(state.buf_state)
  
  def body(state: LoopState):
    key, run_key = jr.split(state.key)
    buf_state = run_episode(run_key, model, model_params, model_state, buf, state.buf_state, env, max_episode_len)
    return LoopState(key, buf_state)

  state = LoopState(key, buf_state)
  state = jax.lax.while_loop(cond, body, state)
  return state.buf_state
