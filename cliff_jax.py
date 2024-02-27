import typing
import jax, jax.numpy as jnp

class Cliff(typing.NamedTuple):
  start_pos: jax.Array # player start position
  end_pos: jax.Array # player end position
  # quadratic cliff dimensions (both are inclusive)
  min_cliff: jax.Array
  max_cliff: jax.Array
  rs: "Cliff.RewardScheme"
  width: int = 8
  height: int = 3
  max_steps: int = 20

  ACTION2DIR = [[-1,0], [1,0], [0,1], [0,-1]]
  ACTION2CHAR = {0: "^", 1: "v", 2: ">", 3: "<"}

  class InternalState(typing.NamedTuple):
    state: jax.Array # int
    step: jax.Array = jnp.asarray(0) # int
  
  class RewardScheme(typing.NamedTuple):
    step_penalty: jax.Array # int
    cliff_penalty: jax.Array # int
    win_reward: jax.Array # int
    not_in_bounds_penalty: jax.Array # int
    #done_fn: typing.Callable # (has_won, has_time, in_bounds, in_cliff): (bool,) * 4 -> bool

    @classmethod
    def full(cls) -> "Cliff.RewardScheme":
      return cls(
        step_penalty=jnp.asarray(-1),
        cliff_penalty=jnp.asarray(-4),
        win_reward=jnp.asarray(51),
        not_in_bounds_penalty=jnp.asarray(0),
        #done_fn=lambda has_won, has_time, _in_bounds, in_cliff: has_won | ~has_time | in_cliff,
      )

  def observe(self, internal_state: "InternalState") -> jax.Array: # int
    return internal_state.state
  
  def step(self, internal_state: "InternalState", action: int) -> tuple["InternalState", float, bool]:
    # compute proposed step result
    loc = self.state2pos(internal_state.state)
    move = jnp.array(self.ACTION2DIR)[action]
    max_pos = jnp.array([self.height - 1, self.width - 1])
    prop_loc = jnp.minimum(jnp.maximum(loc + move, jnp.zeros_like(loc)), max_pos)
    step = internal_state.step + 1

    # compute propositions
    in_bounds = jnp.any(prop_loc != loc)
    in_cliff = jnp.all((self.min_cliff <= prop_loc) & (prop_loc <= self.max_cliff))
    has_won = jnp.all(prop_loc == self.end_pos)
    has_time = jnp.bool(step < self.max_steps)

    # check if step is valid and compute new state
    done = has_won | ~has_time
    newloc = jnp.where(in_bounds & ~in_cliff, prop_loc, loc)
    newstate = self.pos2state(newloc)
    newstep = jnp.where(~done, step, step - 1)

    # compute reward
    rs = self.rs
    reward = (
      rs.step_penalty
      + jnp.where(~in_bounds, rs.not_in_bounds_penalty, 0)
      + jnp.where(in_cliff, rs.cliff_penalty, 0)
      + jnp.where(has_won, rs.win_reward, 0)
    )

    # return new internal state
    return Cliff.InternalState(state=newstate, step=newstep), reward, done
  
  def reset(self) -> "InternalState":
    state = self.pos2state(self.start_pos)
    return Cliff.InternalState(state)
  
  def state2pos(self, state):
    "int -> (y, x): tuple[int, int]"
    return jnp.stack(jnp.divmod(state, self.width))
  
  def pos2state(self, pos):
    "(y, x): tuple[int, int] -> int"
    return pos[0] * self.width + pos[1]
  
  @classmethod
  def full(cls):
    return cls(
      start_pos=jnp.array([2,0]),
      end_pos=jnp.array([2,7]),
      min_cliff=jnp.array([2,1]),
      max_cliff=jnp.array([2,6]),
      rs=Cliff.RewardScheme.full(),
    )
  