import torch
import torch.nn as nn

class Value(nn.Module):
  def __init__(self, num_arms):
    super(Value, self).__init__()
    self.is_recurrent = False
    self.saved_log_probs = []
    self.num_arms = num_arms

  def forward(self, x):
    return 1


class GRU_Value(Value):
  def __init__(self, num_arms, init_state, input_size = 1, hidden_size = 256):
    super(GRU_Value, self).__init__(num_arms)
    self.is_recurrent = True
    self.hidden_size = hidden_size
    self.init_state = init_state

    self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
    self.value = nn.Linear(hidden_size, 1)

    self.prev_state = self.init_state


  def forward(self, x):
    print('states')
    print(x)
    x, h = self.gru(x, self.prev_state)
    self.prev_state = h
    print(x)
    print(h)
    return self.value(x)


  def reset_hidden_state(self):
    self.prev_state = self.init_state