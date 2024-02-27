import typing
import jax, jax.numpy as jnp

class Cliff(typing.NamedTuple):
  start_pos: jax.Array
  end_pos: jax.Array
  # quadratic cliff dimensions (both are inclusive)
  min_cliff: jax.Array
  max_cliff: jax.Array
  width: int = 8
  height: int = 5
  max_steps: int = 20

  ACTION2CHAR = {0: "^", 1: "v", 2: "<", 3: ">"}

  class InternalState(typing.NamedTuple):
    state: int
    step: int = 0

  def observe(self, internal_state: "InternalState") -> int:
    return internal_state.state
  
  def step(self, internal_state: "InternalState", action: int) -> tuple["InternalState", float, bool]:
    ACTION2DIR = jnp.array([[-1,0], [1,0], [0,1], [0,-1]])

    # compute proposed step result
    loc = self.state2pos(internal_state.state)
    move = ACTION2DIR[action]
    max_pos = jnp.array([self.height - 1, self.width - 1])
    prop_loc = jnp.minimum(jnp.maximum(loc + move, jnp.zeros_like(loc)), max_pos)
    step = internal_state.step + 1

    # compute propositions
    in_bounds = jnp.all(prop_loc == loc)
    in_cliff = jnp.all((self.min_cliff <= prop_loc) & (prop_loc <= self.max_cliff))
    has_won = jnp.all(prop_loc == self.end_pos)
    has_time = jnp.bool(step < self.max_steps)

    # check if step is valid
    done = has_won | ~has_time | in_cliff
    newloc = jnp.where(in_bounds & ~in_cliff, prop_loc, loc)
    newstate = self.pos2state(newloc)
    newstep = jnp.where(~done, step, step - 1)
    reward = -1 + jnp.where(in_cliff, -49, 0) + jnp.where(has_won, 51, 0)
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
  
  def default():
    return Cliff(
      start_pos=jnp.array([2,0]),
      end_pos=jnp.array([2,7]),
      min_cliff=jnp.array([2,1]),
      max_cliff=jnp.array([2,6]),
    )
  