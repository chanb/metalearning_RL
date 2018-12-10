import torch
import torch.nn as nn
import math
from helper.values.value import Value
from helper.snail_blocks import TCBlock, AttentionBlock


class SNAILValue(Value):
  # K arms, trajectory of length N
  def __init__(self, output_size, max_num_traj, max_traj_len, encoder, encoder_hidden_size=32, hidden_size=16):
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


  def forward(self, x, hidden_state):
    x = x.transpose(0, 1)  
    x = torch.cat((hidden_state[:, 1:(self.T), :], x), 1)
    next_hidden_state = x

    x = self.encoder(x)
    x = self.value_encoder(x)
    x = self.tc_1(x)
    x = self.tc_2(x)
    x = self.attention_1(x)
    x = self.affine_2(x)
    x = x[:, self.T-1, :].squeeze()  # pick_last_action
    return x.unsqueeze(0).unsqueeze(0), next_hidden_state
