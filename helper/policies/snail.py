import torch
import torch.nn as nn
import numpy as np
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
    def __init__(self, output_size, traj_len, input_size=1, hidden_size=32):
        super(SNAILPolicy, self).__init__(input_size, output_size)
        self.K = output_size
        self.N = traj_len
        self.hidden_size = hidden_size

        num_filters = int(math.floor(math.log(output_size * traj_len + 1)))
        num_channels = output_size + 2 #add 2 because we're adding observation and rewards

        #self.affine_1 = LinearEmbedding(num_channels, hidden_size)
        #num_channels += hidden_size

        self.tc_1 = TCBlock(num_channels, self.N, hidden_size)
        num_channels += num_filters * hidden_size

        self.tc_2 = TCBlock(num_channels, self.N, hidden_size)
        num_channels += num_filters * hidden_size

        self.attention_1 = AttentionBlock(num_channels, hidden_size, hidden_size)
        num_channels += hidden_size

        self.affine_2 = nn.Linear(num_channels, self.K)

    def forward(self, states, actions, rewards):
        actions_onehot = torch.Tensor(np.zeros((len(actions), self.K, 1)))# convert actions to one-hot encoding
        states_tensor = torch.Tensor(np.expand_dims(np.expand_dims(states, 1), 1))
        rewards_tensor = torch.Tensor(np.expand_dims(np.expand_dims(rewards, 1), 1))
        x = torch.cat((states_tensor, actions_onehot, rewards_tensor), 1)
        x = torch.transpose(x, 1, 2)
        x = self.tc_1(x)
        x = self.tc_2(x)
        x = self.attention_1(x)
        x = self.affine_2(x)
        res1 = F.softmax(x.squeeze(), dim=1)
        return res1
