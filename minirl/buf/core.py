import jax

class Buf:
  # change buffer contents
  def append(self, state): ...
  def end_episode(self, state): ...
  def empty(self): ...
  def fill(self, state, key, model, model_params, model_state, env): ...

  # RL related buffer interactions
  def get_episode_reward(self, state) -> jax.Array: ...
  def reduce_episodes(self, state, fn, carry_init) -> jax.Array: ...
