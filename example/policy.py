import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import *

# Generic policy
class Policy(nn.Module):
  def __init__(self, num_arms):
    super(Policy, self).__init__()
    self.is_recurrent = False
    self.saved_log_probs = []
    self.num_arms = num_arms
  

  def forward(self, x):
    return torch.tensor(1/self.num_arms).repeat(self.num_arms)


# Fully Connected Network
class FCN_Policy(Policy):
  def __init__(self, num_arms):
    super(FCN_Policy, self).__init__(num_arms)

    self.affine_1 = nn.Linear(1, 128)
    self.affine_2 = nn.Linear(128, num_arms)


  def forward(self, x):
    x = F.relu(self.affine_1(x))
    action_scores = self.affine_2(x)
    return F.softmax(action_scores, dim=1)


# SNAIL
class SNAIL_Policy(Policy):
  # K arms, trajectory of length N
  def __init__(self, num_arms, traj_len):
    super(SNAIL_Policy, self).__init__(num_arms)
    self.K = num_arms
    self.N = traj_len

    num_filters = int(math.ceil(math.log(num_arms * traj_len + 1)))
    num_channels = num_arms + 1

    self.affine_1 = nn.Linear(num_channels, 32)
    num_channels += 32
    
    self.tc_1 = TCBlock(num_channels, traj_len, 32)
    num_channels += num_filters * 32

    self.tc_2 = TCBlock(num_channels, traj_len, 32)
    num_channels += num_filters * 32

    self.attention_1 = AttentionBlock(num_channels, 32, 32)
    num_channels += 32

    self.affine_2 = nn.Linear(num_channels, num_arms)

  def forward(self, x):
    x = self.affine_1(x)
    x = self.tc_1(x)
    x = self.tc_2(x)
    x = self.attention_1(x)
    x = self.affine_2(x)
    return F.softmax(x, dim=1)


# GRU
class GRU_Policy(Policy):
  def __init__(self, num_arms, init_state, input_size = 1, hidden_size = 256):
    super(GRU_Policy, self).__init__(num_arms)
    self.is_recurrent = True
    self.hidden_size = hidden_size
    self.init_state = init_state

    self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
    self.affine = nn.Linear(hidden_size, num_arms)

    self.prev_state = self.init_state


  def forward(self, x):
    x, h = self.gru(x, self.prev_state)
    self.prev_state = h
    x = self.affine(x).squeeze(0)
    return F.softmax(x, dim=1)


  def reset_hidden_state(self):
    self.prev_state = self.init_state