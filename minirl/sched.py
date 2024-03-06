import jax, jax.numpy as jnp, math, dataclasses, typing

class Schedule:
  def step(self, params):
    # increase current step
    t = params
    return t + 1

  def init(self):
    return jnp.asarray(0, dtype=int)
  
@dataclasses.dataclass(frozen=True)
class ExpSchedule(Schedule):
  r"Exponential schedule yielding numbers according to $a\cdot\exp(bt)+c$ where $t$ is guided by the parameter."
  a: float # scale of exponential function
  b: float # weight of x before exp-ing it
  c: float # offset from x-axis

  def __call__(self, params) -> jax.Array:
    t = params
    return self.a * jnp.exp(self.b * t) + self.c

  @classmethod
  def from_points(cls, x1: tuple[float, float], x2: tuple[float, float]):
    b = math.log(x1[1] / x2[1]) / (x1[0] - x2[0])
    return cls(a=x2[0] / math.exp(b * x1[0]), b=b, c=0.)

@dataclasses.dataclass(frozen=True)
class LinearSchedule(Schedule):
  r"Schedule defined by a line that optionally stays within bounds."
  a: float # slope of the line
  b: float # offset of the line
  l: typing.Optional[float] = None # optional lowest value
  h: typing.Optional[float] = None # optional highest value

  def __call__(self, params) -> jax.Array:
    t = params
    v = self.a * t + self.b
    if self.l is not None:
      v = jnp.maximum(v, self.l)
    if self.h is not None:
      v = jnp.minimum(v, self.h)
    return v
  
  @classmethod
  def bounded_within(cls, x1: float, x2: float, y1: float, y2: float):
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    l, h = min(y1, y2), max(y1, y2)
    return cls(a, b, l, h)
    