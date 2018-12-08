import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from helper.policies.policy import Policy
from helper.snail_blocks import *

class LinearEmbedding(Policy):
  def __init__(self, input_size=1, output_size=32):
    super(LinearEmbedding, self).__init__(input_size, output_size)
    self.fcn = nn.Linear(input_size, output_size)

  def forward(self, x):
    return self.fcn(x)


class SNAILPolicy(Policy):
  # K arms, trajectory of length N
  def __init__(self, output_size, max_num_traj, max_traj_len, encoder, input_size=1, hidden_size=32, num_traj=1):
    super(SNAILPolicy, self).__init__(input_size, output_size)
    self.K = output_size
    self.N = max_num_traj
    self.T = max_num_traj * max_traj_len
    self.hidden_size = hidden_size
    self.is_recurrent = True
    
    num_channels = 0

    self.encoder = encoder
    num_channels += hidden_size

    num_filters = int(math.ceil(math.log(self.T)))

    self.tc_1 = TCBlock(num_channels, self.T, hidden_size)
    num_channels += num_filters * hidden_size

    self.tc_2 = TCBlock(num_channels, self.T, hidden_size)
    num_channels += num_filters * hidden_size

    self.attention_1 = AttentionBlock(num_channels, hidden_size, hidden_size)
    num_channels += hidden_size

    self.affine_2 = nn.Linear(num_channels, self.K)


  def forward(self, x, hidden_state, to_print=True):
    x = x.transpose(0, 1)  
    x = torch.cat((hidden_state[:, 1:(self.T), :], x), 1)
    next_hidden_state = x

    x = self.encoder(x) # result: traj_len x 32
    x = self.tc_1(x)
    x = self.tc_2(x)
    x = self.attention_1(x)
    x = self.affine_2(x)
    print('MOVES: {}'.format(x))
    x = x[:, self.T-1, :] # pick_last_action
    if (to_print):
      print('Distribution: {}'.format(F.softmax(x, dim=1)))
    return Categorical(logits=x), next_hidden_state
