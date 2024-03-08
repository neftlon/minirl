import pytest
from minirl.corridor import Corridor

def test_finishing_seq():
  env = Corridor()
  state = env.reset(())
  for action in (1,0,1):
    state, reward, done = env.step(state, action)
    assert reward == -1
  assert state == 3
  assert done

if __name__ == "__main__":
  pytest.main([__file__])