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
    def __init__(self, output_size, traj_len, encoder, input_size=1, hidden_size=32):
        super(SNAILPolicy, self).__init__(input_size, output_size)
        self.K = output_size
        self.N = traj_len
        self.hidden_size = hidden_size

        num_channels = 0

        self.encoder = encoder
        num_channels += hidden_size

        num_filters = int(math.floor(math.log(output_size * traj_len + 1)))

        self.tc_1 = TCBlock(num_channels, self.N, hidden_size)
        num_channels += num_filters * hidden_size

        self.tc_2 = TCBlock(num_channels, self.N, hidden_size)
        num_channels += num_filters * hidden_size

        self.attention_1 = AttentionBlock(num_channels, hidden_size, hidden_size)
        num_channels += hidden_size

        self.affine_2 = nn.Linear(num_channels, self.K)

    def forward(self, observations, actions, rewards):
        # observations: 2-dim array with nobs x 2 (state, done)
        # actions: array with nobs elements
        # rewards: array with nobs element
        if actions.size > 0:
            actions_onehot = np.eye(self.K)[actions.astype(int)] # nobs x num_action
            rewards = np.expand_dims(rewards, 1) # nobs x 1
            x = np.hstack((observations, actions_onehot, rewards))
            x = np.vstack((np.zeros((self.N - observations.shape[0], x.shape[1])), x)) # pad x with 0s
        else: # no actions and rewards yet
            x = np.zeros((self.N, self.K + 3))
            if observations.size > 0: # if we already observe the first state
                x[self.N-1, 1:2] = observations[0, :]
        x = torch.from_numpy(x).float()
        x = self.encoder(x) # result: traj_len x 32
        x = x.unsqueeze(1) # add a new dimension at the second dim
        x = self.tc_1(x)
        x = self.tc_2(x)
        x = self.attention_1(x)
        x = self.affine_2(x)
        x = x[self.N-1, :, :] # pick_last_action
        res1 = F.softmax(x, dim=1)
        return res1
