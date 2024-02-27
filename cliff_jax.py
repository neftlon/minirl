import typing, dataclasses, chex
import jax, jax.numpy as jnp, jax.random as jr
from util import Env

class Rect(typing.NamedTuple):
  # both are inclusive
  min: tuple[int, int]
  max: tuple[int, int]

  def contains(self, p: jax.Array) -> jax.Array:
    min, max = jnp.asarray(self.min), jnp.asarray(self.max)
    return jnp.all((min <= p) & (p <= max))

@dataclasses.dataclass(frozen=True)
class Cliff(Env):
  start_pos: typing.Optional[tuple[int, int] | typing.Sequence[tuple[int, int]]] # player start position
  end_pos: typing.Optional[tuple[int, int]] # player end position
  # rectangular cliff location
  cliff: typing.Optional[Rect]
  # reward scheme
  rs: "Cliff.RewardScheme"
  width: int = 8
  height: int = 3
  max_steps: int = 20

  ACTION2DIR = [[-1,0], [1,0], [0,1], [0,-1]]
  ACTION2CHAR = {0: "^", 1: "v", 2: ">", 3: "<"}

  @chex.dataclass(frozen=True)
  class InternalState(Env.InternalState):
    state: jax.Array # int
    step: jax.Array # int

    @classmethod
    def start(cls, state) -> "Cliff.InternalState":
      return cls(state=state, step=jnp.asarray(0))
  
  class RewardScheme(typing.NamedTuple):
    step_penalty: int
    cliff_penalty: int
    win_reward: int
    not_in_bounds_penalty: int
    # (has_won, has_time, in_bounds, in_cliff)
    done_pos_mask: tuple[bool, bool, bool, bool] # (4,)
    done_neg_mask: tuple[bool, bool, bool, bool] # (4,)

    @classmethod
    def full(cls) -> "Cliff.RewardScheme":
      return cls(
        step_penalty=-1,
        cliff_penalty=-4,
        win_reward=51,
        not_in_bounds_penalty=0,
        done_pos_mask=(True,False,False,False),
        done_neg_mask=(False,True,False,False),
      )

  def observe(self, internal_state: "InternalState") -> jax.Array: # int
    return internal_state.state
  
  def step(self, internal_state: "InternalState", action: int) -> Env.StepResult:
    rs = self.rs

    # compute proposed step result
    loc = self.state2pos(internal_state.state)
    move = jnp.array(self.ACTION2DIR)[action]
    max_pos = jnp.array([self.height - 1, self.width - 1])
    prop_loc = jnp.minimum(jnp.maximum(loc + move, jnp.zeros_like(loc)), max_pos)
    step = internal_state.step + 1

    # compute propositions
    in_bounds = jnp.any(prop_loc != loc)
    in_cliff = self.cliff.contains(prop_loc) if self.cliff is not None else jnp.asarray(False)
    has_won = jnp.all(prop_loc == jnp.asarray(self.end_pos)) if self.end_pos is not None else jnp.asarray(False)
    has_time = jnp.bool(step < self.max_steps)

    # check if step is valid and compute new state
    bools = jnp.stack((has_won, has_time, in_bounds, in_cliff))
    done = jnp.any((bools & jnp.asarray(rs.done_pos_mask)) | (~bools & jnp.asarray(rs.done_neg_mask)))
    newloc = jnp.where(in_bounds & ~in_cliff, prop_loc, loc)
    newstate = self.pos2state(newloc)
    newstep = jnp.where(~done, step, step - 1)

    # compute reward
    reward = (
      rs.step_penalty
      + jnp.where(~in_bounds, rs.not_in_bounds_penalty, 0)
      + jnp.where(in_cliff, rs.cliff_penalty, 0)
      + jnp.where(has_won, rs.win_reward, 0)
    )

    # return new internal state
    newinternalstate = Cliff.InternalState(state=newstate, step=newstep)
    return Env.StepResult(state=newinternalstate, reward=reward, done=done)
  
  def reset(self, key) -> "InternalState":
    if self.start_pos is None:
      # pick a random position on whole field
      state = jr.choice(key, 24)
    else:
      start_pos = jnp.asarray(self.start_pos)
      assert start_pos.ndim in [1,2]
      if start_pos.ndim == 2:
        # pick a random position from the given ones
        start_pos = jr.choice(key, start_pos)
      state = self.pos2state(start_pos)
    return Cliff.InternalState.start(state)
  
  def state2pos(self, state):
    "int -> (y, x): tuple[int, int]"
    return jnp.stack(jnp.divmod(state, self.width))
  
  def pos2state(self, pos):
    "(y, x): tuple[int, int] -> int"
    return pos[0] * self.width + pos[1]
  
  @classmethod
  def full(cls) -> "Cliff":
    return cls(
      start_pos=(2,0),
      end_pos=(2,7),
      cliff=Rect(min=(2,1), max=(2,6)),
      rs=Cliff.RewardScheme.full(),
    )
  