import torch
import torch.nn as nn
from helper.values import GRUValue
from helper.policies import GRUPolicy
import torch.nn.functional as F
from torch.distributions import Categorical
from helper.model_init import weight_init


class GRUActorCritic(nn.Module):
  def __init__(self, output_size, input_size=1, hidden_size=256, non_linearity='none'):
    super(GRUActorCritic, self).__init__()
    self.is_recurrent = True
    self.hidden_size = hidden_size

    self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
    self.relu1 = nn.ReLU()
    self.policy = nn.Linear(hidden_size, output_size)
    self.value = nn.Linear(hidden_size, 1)
    if non_linearity == 'sigmoid':
      self.non_linearity = nn.Sigmoid()
    elif non_linearity == 'tanh':
      self.non_linearity = nn.Tanh()
    elif non_linearity == 'relu':
      self.non_linearity = nn.ReLU()
    else:
      self.non_linearity = None
    # self.apply(weight_init)

  #TODO: Remove to_print
  def forward(self, x, h, to_print=True):
    x, h = self.gru(x, h)
    x = self.relu1(x)
    val = self.value(x)
    if (self.non_linearity):
      val = self.non_linearity(val)
    dist = self.policy(x).squeeze(0)
    
    if (to_print):
      print('Distribution: {}, Val: {}'.format(F.softmax(dist, dim=1), val))

    return Categorical(logits=dist), val, h

  def init_hidden_state(self):
    return torch.zeros([1, 1, self.hidden_size])