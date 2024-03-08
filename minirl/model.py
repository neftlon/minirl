import jax, jax.numpy as jnp, jax.random as jr, typing
from .sched import ExpSchedule

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

  def __call__(self, model_params, _model_state, key, x) -> jax.Array:
    key, eps_key = jr.split(key)
    # use cond to run the model only when it's necessary
    return jax.lax.cond(
      jr.uniform(eps_key) >= self.eps,
      # take model's prediction (greedy)
      lambda: self.logits_model(model_params, x).argmax(axis=-1),
      # select a random action
      lambda: jr.choice(key, self.num_actions),
    )
  
  def logp(self, model_params, _model_state, x, y):
    # compute prediction's logp
    logits = self.logits_model(model_params, x)
    model_logp = jax.nn.log_softmax(logits)[y]
    # weight logp according to action selection strategy
    return jnp.log(self.eps / self.num_actions + (1 - self.eps) * jnp.exp(model_logp))

  def step(self, _params):
    ...

  def init(self, key):
    return self.logits_model.init(key), ()

class TemperatureSampler(typing.NamedTuple):
  logits_model: typing.Any
  sched: ExpSchedule

  def __call__(self, model_params, model_state, key, x) -> jax.Array:
    sched_state = model_state

    # compute logits
    logits = self.logits_model(model_params, x)

    # scale logits by temperature tau
    tau = self.sched(sched_state) # get temperature from schedule
    logits = logits / tau

    # sample action from logits
    return jr.categorical(key, logits)
  
  def logp(self, model_params, model_state, x, y):
    sched_state = model_state

    logits = self.logits_model(model_params, x)
    
    # weight logits by tau
    tau = self.sched(sched_state) # get temperature from schedule
    logits = logits / tau

    # extract log-probability for action
    return jax.nn.log_softmax(logits)[y]
  
  def step(self, model_state):
    sched = model_state
    return self.sched.step(sched)
  
  def init(self, key):
    model_params = self.logits_model.init(key)
    model_state = self.sched.init()
    return model_params, model_state
