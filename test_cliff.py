import pytest
from cliff import Cliff

@pytest.mark.parametrize("oob_state", [-1, 24, 1000, -100])
def test_state_out_of_bounds(oob_state):
  assert Cliff().state_to_pos(oob_state) == Cliff.player_init()

def test_empty_pack():
  # packing an empty field should reset the player to its initial location
  env = Cliff()
  field = env.empty()
  init_state = env.reset()
  assert init_state == env.pack(field)

def test_move_to_finish():
  env = Cliff()
  state = env.pos_to_state((1, 7)) # one above finish
  assert state == env.width * 1 + 7

  win_state = env.pos_to_state(Cliff.player_win())
  assert (win_state, 20, True) == env.step(state, 1)

if __name__ == "__main__":
  pytest.main([__file__])
