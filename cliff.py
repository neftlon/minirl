import dataclasses
import torch

@dataclasses.dataclass
class Cliff:
  "simple environment for cliff"
  current_step: int = 0 # internal state
  max_steps: int = 20
  width: int = 8
  height: int = 3

  MOVE_NAME_AND_DIR = {
    0: ("^", torch.as_tensor([-1,0])), 1: ("v", torch.as_tensor([1,0])),
    2: (">", torch.as_tensor([0,1])), 3: ("<", torch.as_tensor([0,-1])),
  }

  def step(self, state: int, action: int) -> tuple[int, float, bool]:
    """
    Returns:
      state: The (observable) state of the environment after taking action
      reward: This step's reward
      done: If the environment is finished (either truncated or done)
    """
    assert 0 <= action < 4, "action out of bounds"
    loc = torch.as_tensor(self.state_to_pos(state)) # unpack state
    move = self.MOVE_NAME_AND_DIR[action][1] # extract dir from move and dir
    max_pos = torch.as_tensor([self.height - 1, self.width - 1])
    prop_loc = torch.minimum(torch.maximum(loc + move, torch.as_tensor(0)), max_pos)
    out_of_bounds = torch.all(loc == prop_loc)
    if out_of_bounds:
      return state, -1, self.current_step >= self.max_steps

    in_cliff = torch.all((prop_loc[0] == 2) & (1 <= prop_loc[1]) & (prop_loc[1] <= 6))
    if in_cliff:
      return state, -100, True # terminate after walking into cliff
    
    won = torch.all(prop_loc == torch.as_tensor(Cliff.player_win()))
    prop_loc = self.pos_to_state(prop_loc).item() # encode location proposal
    if won:
      return prop_loc, 20, True # player is in win location
    # accept move
    self.current_step += 1
    return prop_loc, -1, (self.current_step > self.max_steps)

  def reset(self) -> int:
    self.current_step = 0 # reset step variable
    return self.pos_to_state(Cliff.player_init())

  def pos_to_state(self, pos):
    y, x = pos
    return y * self.width + x
  
  def state_to_pos(self, state):
    if 0 <= state < 24: # state should be in bounds
      y, x = divmod(state, self.width)
      return y, x
    
    # default to initial location
    return Cliff.player_init() 
  
  def empty(self):
    field = torch.zeros((self.height, self.width))
    for x in range(1,7):
      field[2,x] = 1
    return field

  def player_init():
    "returns initial location of player"
    return 2, 0
  
  def player_win():
    "Returns location of win"
    return 2, 7

  def unpack(self, state):
    field = self.empty()
    y, x = self.state_to_pos(state)
    field[y, x] = 2
    return field

  def pack(self, field):
    for state in range(0,24):
      y, x = self.state_to_pos(state)
      if field[y, x] == 2:
        return self.pos_to_state((y, x))
    # default to player init
    return self.pos_to_state(Cliff.player_init())
