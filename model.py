import jax, jax.numpy as jnp, jax.random as jr
import typing

class Embedding(typing.NamedTuple):
  num_emb: int
  emb_dim: int

  def __call__(self, params, _key, x):
    return params[x]
  
  def logp(self, params, x, y):
    logits = self(params, ..., x)
    return jax.nn.log_softmax(logits)[y]

  def init(self, key):
    return jr.normal(key, (self.num_emb, self.emb_dim))

class Mlp(typing.NamedTuple):
  sizes: tuple[int, ...]
  activation: typing.Callable[[jax.Array], jax.Array] = jax.nn.leaky_relu
  bias_mode: str = "zeros"

  def __call__(self, params, _key, x):
    for w, b in params[:-1]:
      x = jnp.dot(x, w) + b
      x = self.activation(x)
    w, b = params[-1]
    return jnp.dot(x, w) + b
  
  def logp(self, params, x, y):
    logits = self(params, ..., x)
    return jax.nn.log_softmax(logits)[y]
  
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
  model: typing.Any
  num_actions: int
  eps: float = .1

  def __call__(self, params, key, *args):
    key, eps_key = jr.split(key)
    # use cond to run the model only when it's necessary
    return jax.lax.cond(
      jr.uniform(eps_key) >= self.eps,
      # take model's prediction (greedy)
      lambda: self.model(params, key, *args).argmax(axis=-1),
      # select a random action
      lambda: jr.choice(key, self.num_actions),
    )
  
  def logp(self, params, x, y):
    return jnp.log(self.eps / self.num_actions + (1 - self.eps) * jnp.exp(self.model.logp(params, x, y)))

  def init(self, key):
    return self.model.init(key)