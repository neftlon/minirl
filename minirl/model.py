import jax, jax.numpy as jnp, jax.random as jr
import typing, math

class Embedding(typing.NamedTuple):
  num_emb: int
  emb_dim: int

  def __call__(self, params, x):
    assert params.shape == (self.num_emb, self.emb_dim)
    return params[x]

  def init(self, key):
    return jr.normal(key, (self.num_emb, self.emb_dim))

class Mlp(typing.NamedTuple):
  sizes: tuple[int, ...]
  activation: typing.Callable[[jax.Array], jax.Array] = jax.nn.leaky_relu
  bias_mode: str = "zeros"

  def __call__(self, params, x):
    for w, b in params[:-1]:
      x = jnp.dot(x, w) + b
      x = self.activation(x)
    w, b = params[-1]
    return jnp.dot(x, w) + b
  
  def init(self, key):
    if self.bias_mode == "zeros":
      keys = jr.split(key, len(self.sizes) - 1)
      z = zip(keys, self.sizes[:-1], self.sizes[1:])
      return [(jr.normal(key, (m, n)), jnp.zeros(n)) for key, m, n in z]
    elif self.bias_mode == "random":
      keys = jr.split(key, (len(self.sizes) - 1, 2))
      z = zip(keys, self.sizes[:-1], self.sizes[1:])
      return [(jr.normal(w, (m, n)), jr.normal(b, (n,))) for (w, b), m, n in z]
    raise ValueError("invalid bias mode")

class EpsGreedy(typing.NamedTuple):
  logits_model: typing.Any
  num_actions: int
  eps: float = .1

  def __call__(self, params, key, x):
    key, eps_key = jr.split(key)
    # use cond to run the model only when it's necessary
    return jax.lax.cond(
      jr.uniform(eps_key) >= self.eps,
      # take model's prediction (greedy)
      lambda: self.logits_model(params, x).argmax(axis=-1),
      # select a random action
      lambda: jr.choice(key, self.num_actions),
    )
  
  def logp(self, params, x, y):
    # compute prediction's logp
    logits = self.logits_model(params, x)
    model_logp = jax.nn.log_softmax(logits)[y]
    # weight logp according to action selection strategy
    return jnp.log(self.eps / self.num_actions + (1 - self.eps) * jnp.exp(model_logp))

  def step(self, _params):
    ...

  def init(self, key):
    return self.logits_model.init(key)

class ExpSchedule(typing.NamedTuple):
  r"Exponential schedule yielding numbers according to $a\cdot\exp(bt)+c$ where $t$ is guided by the parameter."
  a: float # scale of exponential function
  b: float # weight of x before exp-ing it
  c: float # offset from x-axis

  def __call__(self, params) -> jax.Array:
    t = params
    return self.a * jnp.exp(self.b * t) + self.c

  def step(self, params):
    # increase current step
    t = jax.lax.stop_gradient(params) # disallow gradient computation on t
    return t + 1

  def init(self):
    return jnp.asarray(0, dtype=int)

  @classmethod
  def from_points(cls, x1: tuple[float, float], x2: tuple[float, float]):
    b = math.log(x1[1] / x2[1]) / (x1[0] - x2[0])
    return cls(a=x2[0] / math.exp(b * x1[0]), b=b, c=0.)

class TemperatureSampler(typing.NamedTuple):
  logits_model: typing.Any
  sched: ExpSchedule

  def __call__(self, model_params, model_state, x) -> jax.Array:
    # compute logits
    logits = self.logits_model(model_params, x)

    # scale logits by temperature tau
    tau = self.sched(model_state["sched"]) # get temperature from schedule
    logits = logits / tau

    # sample action from logits
    return jr.categorical(model_state["key"], logits)
  
  def logp(self, model_params, model_state, x, y):
    logits = self.logits_model(model_params, x)
    
    # weight logits by tau
    tau = self.sched(model_state["sched"]) # get temperature from schedule
    logits = logits / tau

    # extract log-probability for action
    return jax.nn.log_softmax(logits)[y]
  
  def step(self, model_state):
    sched, key = model_state["sched"], model_state["key"]
    return {"sched": self.sched.step(sched), "key": jr.split(key)[0]}
  
  def init(self, key):
    model_key, state_key = jr.split(key)
    model_params = self.logits_model.init(model_key)
    model_state = {"sched": self.sched.init(), "key": state_key}
    return model_params, model_state
