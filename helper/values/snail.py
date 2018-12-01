import torch
import torch.nn as nn
import numpy as np
from helper.values.value import Value
from helper.snail_blocks import *


class SNAILValue(Value):
  # K arms, trajectory of length N
  def __init__(self, output_size, max_num_traj, max_traj_len, encoder, input_size=1, encoder_hidden_size=32, hidden_size=16, non_linearity='none'):
    super(SNAILValue, self).__init__(output_size)
    self.K = output_size
    self.N = max_num_traj
    self.T = max_num_traj * max_traj_len
    self.hidden_size = hidden_size
    self.is_recurrent = True

    num_channels = 0

    self.encoder = encoder
    self.value_encoder = nn.Linear(encoder_hidden_size, hidden_size)

    num_channels += hidden_size

    num_filters = int(math.ceil(math.log(self.T)))

    self.tc_1 = TCBlock(num_channels, self.T, hidden_size)
    num_channels += num_filters * hidden_size

    self.tc_2 = TCBlock(num_channels, self.T, hidden_size)
    num_channels += num_filters * hidden_size

    self.attention_1 = AttentionBlock(num_channels, hidden_size, hidden_size)
    num_channels += hidden_size

    self.affine_2 = nn.Linear(num_channels, 1)

    if non_linearity == 'sigmoid':
      self.non_linearity = nn.Sigmoid()
    elif non_linearity == 'tanh':
      self.non_linearity = nn.Tanh()
    elif non_linearity == 'relu':
      self.non_linearity = nn.ReLU()
    else:
      self.non_linearity = None


  def forward(self, x, hidden_state):
    if hidden_state.size()[0] == 0:
      x = x
    elif hidden_state.shape[0] >= self.T:
      x = torch.cat((hidden_state[1:(self.T), :, :], x))
    else:
      x = torch.cat((hidden_state, x))

    next_hidden_state = x

    if x.shape[0] < self.T:
      x = torch.cat((torch.FloatTensor(self.T - x.shape[0], x.shape[1], x.shape[2]).zero_(), x))

    x = self.encoder(x)
    x = self.value_encoder(x)
    x = self.tc_1(x)
    x = self.tc_2(x)
    x = self.attention_1(x)
    x = self.affine_2(x)
    x = x[self.T-1, :, :].squeeze()  # pick_last_action
    if (self.non_linearity):
      x = self.non_linearity(x)
    return x.unsqueeze(0).unsqueeze(0), next_hidden_state
