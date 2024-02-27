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