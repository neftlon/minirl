import typing, dataclasses, chex
import jax, jax.numpy as jnp, jax.random as jr
from .util import Env

class Rect(typing.NamedTuple):
  # both are inclusive
  min: tuple[int, int]
  max: tuple[int, int]

  def contains(self, p: jax.Array) -> jax.Array:
    min, max = jnp.asarray(self.min), jnp.asarray(self.max)
    return jnp.all((min <= p) & (p <= max))

@dataclasses.dataclass(frozen=True)
class Cliff(Env):
  class RewardScheme(typing.NamedTuple):
    step_penalty: int = -1
    cliff_penalty: int = -99 # with step_penalty, this sums to -100
    win_reward: int = 21 # with step_penalty, this sums to 20
    not_in_bounds_penalty: int = -2
    # (has_won, has_time, in_bounds, in_cliff)
    done_pos_mask: tuple[bool, bool, bool, bool] = (True,False,False,False) # (4,)
    done_neg_mask: tuple[bool, bool, bool, bool] = (False,True,False,False)# (4,)
  
  start_pos: typing.Optional[tuple[int, int] | tuple[tuple[int, int], ...]] = (3,0) # player start position
  end_pos: typing.Optional[tuple[int, int]] = (3,11) # player end position
  # rectangular cliff location
  cliff: typing.Optional[Rect] = Rect(min=(3,1), max=(3,10))
  # reward scheme
  reward_scheme: "Cliff.RewardScheme" = RewardScheme()
  width: int = 12
  height: int = 4
  max_steps: int = 50

  ACTION2DIR = [[-1,0], [1,0], [0,1], [0,-1]]
  ACTION2CHAR = {0: "^", 1: "v", 2: ">", 3: "<"}

  @chex.dataclass(frozen=True)
  class InternalState(Env.InternalState):
    state: jax.Array # int
    step: jax.Array # int

    @classmethod
    def start(cls, state) -> "Cliff.InternalState":
      return cls(state=state, step=jnp.asarray(0))

  def observe(self, internal_state: "InternalState") -> jax.Array: # int
    return internal_state.state
  
  def step(self, internal_state: "InternalState", action: int) -> Env.StepResult:
    rs = self.reward_scheme

    # compute proposed step result
    loc = jnp.asarray(Cliff.state2pos(self.width, internal_state.state))
    move = jnp.array(self.ACTION2DIR)[action]
    max_pos = jnp.array([self.height - 1, self.width - 1])
    prop_loc = jnp.minimum(jnp.maximum(loc + move, jnp.zeros_like(loc)), max_pos)
    step = internal_state.step + 1

    # compute propositions
    in_bounds = jnp.any(prop_loc != loc)
    in_cliff = self.cliff.contains(prop_loc) if self.cliff is not None else jnp.asarray(False)
    has_won = jnp.all(prop_loc == jnp.asarray(self.end_pos)) if self.end_pos is not None else jnp.asarray(False)
    has_time = jnp.asarray(step < self.max_steps)

    # check if step is valid and compute new state
    bools = jnp.stack((has_won, has_time, in_bounds, in_cliff))
    done = jnp.any((bools & jnp.asarray(rs.done_pos_mask)) | (~bools & jnp.asarray(rs.done_neg_mask)))
    newloc = jnp.where(in_bounds & ~in_cliff, prop_loc, loc)
    newstate = Cliff.pos2state(self.width, newloc)
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
      state = Cliff.pos2state(self.width, start_pos)
    return Cliff.InternalState.start(state)
  
  @staticmethod
  def state2pos(width, state):
    "int -> (y, x): tuple[int, int]"
    return divmod(state, width)
  
  @staticmethod
  def pos2state(width, pos):
    "(y, x): tuple[int, int] -> int"
    return pos[0] * width + pos[1]
  
  @classmethod
  def avoid_walls(cls, max_steps: int) -> "Cliff":
    return cls(
      start_pos=None, # can start in a random location
      end_pos=None, # there is no end location
      cliff=None, # there is no cliff either
      reward_scheme=Cliff.RewardScheme(
        step_penalty=1, # steps are good
        cliff_penalty=0, # there is no cliff
        win_reward=0, # there is no winning
        not_in_bounds_penalty=-5, # not in bounds is bad
        done_pos_mask=(False,False,False,False),
        done_neg_mask=(False,True,False,False),
      ),
      max_steps=max_steps,
    )
  
  @classmethod
  def reach_goal_without_cliff(cls, max_steps: int) -> "Cliff":
    return cls(
      start_pos=None, # random start position
      cliff=None, # no cliff
      max_steps=max_steps,
    )
  
  @classmethod
  def avoid_walls_and_cliff(cls, max_steps: int) -> "Cliff":
    width = 8
    start_pos = tuple([Cliff.state2pos(width, s) for s in [*range(0,17)] + [23]])
    return cls(
      start_pos=start_pos,
      end_pos=None,
      reward_scheme=Cliff.RewardScheme(
        step_penalty=1,
        cliff_penalty=-5,
        win_reward=0,
        not_in_bounds_penalty=-5,
        done_pos_mask=(False,False,False,False),
        done_neg_mask=(False,True,False,False),
      ),
      max_steps=max_steps,
    )
