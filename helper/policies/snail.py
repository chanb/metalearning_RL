import torch
import torch.nn as nn
from helper.policies.policy import Policy
from helper.snail_blocks import *


class SNAILPolicy(Policy):
    # K arms, trajectory of length N
    def __init__(self, output_size, traj_len, input_size=1):
        super(SNAILPolicy, self).__init__(input_size, output_size)
        self.K = output_size
        self.N = traj_len

        num_filters = int(math.ceil(math.log(output_size * traj_len + 1)))
        num_channels = output_size + 1

        self.affine_1 = nn.Linear(num_channels, 32)
        num_channels += 32

        self.tc_1 = TCBlock(num_channels, traj_len, 32)
        num_channels += num_filters * 32

        self.tc_2 = TCBlock(num_channels, traj_len, 32)
        num_channels += num_filters * 32

        self.attention_1 = AttentionBlock(num_channels, 32, 32)
        num_channels += 32

        self.affine_2 = nn.Linear(num_channels, output_size)

    def forward(self, x):
        x = self.affine_1(x)
        x = self.tc_1(x)
        x = self.tc_2(x)
        x = self.attention_1(x)
        x = self.affine_2(x)
        return F.softmax(x, dim=1)
