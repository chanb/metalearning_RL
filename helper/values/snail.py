import torch
import torch.nn as nn
import numpy as np
from helper.values.value import Value
from helper.snail_blocks import *


class SNAILValue(Value):
    # K arms, trajectory of length N
    def __init__(self, output_size, traj_len, encoder, input_size=1, encoder_hidden_size=32, hidden_size=16, non_linearity='none'):
        super(SNAILValue, self).__init__(output_size)
        self.K = output_size
        self.N = traj_len
        self.hidden_size = hidden_size

        num_channels = 0

        self.encoder = encoder
        self.value_encoder = nn.Linear(encoder_hidden_size, hidden_size)

        num_channels += hidden_size

        num_filters = int(math.floor(math.log(output_size * traj_len + 1)))

        self.tc_1 = TCBlock(num_channels, self.N, hidden_size)
        num_channels += num_filters * hidden_size

        self.tc_2 = TCBlock(num_channels, self.N, hidden_size)
        num_channels += num_filters * hidden_size

        self.attention_1 = AttentionBlock(num_channels, hidden_size, hidden_size)
        num_channels += hidden_size

        self.affine_2 = nn.Linear(num_channels, output_size)

        if non_linearity == 'sigmoid':
            self.non_linearity = nn.Sigmoid()
        elif non_linearity == 'tanh':
            self.non_linearity = nn.Tanh()
        elif non_linearity == 'relu':
            self.non_linearity = nn.ReLU()
        else:
            self.non_linearity = None

        # Keep past information
        self.observations = np.array([])
        self.actions = np.array([])
        self.rewards = np.array([])

    def forward(self, state, action, reward, done, keep=True):
        if self.observations.size == 0:
            observations = np.array([[state[0], done]])
        else:
            observations = np.stack((self.observations, np.array([[state[0], done]])),
                                          axis=0)
        if action != -1:
            actions = np.append(self.actions, action.item())
            rewards = np.append(self.rewards, reward)

        # observations: 2-dim array with nobs x 2 (state, done)
        # actions: array with nobs elements
        # rewards: array with nobs element
        if actions.size > 0:
            actions_onehot = np.eye(self.K)[actions.astype(int)] # nobs x num_action
            rewards = np.expand_dims(rewards, 1) # nobs x 1
            x = np.hstack((observations, actions_onehot, rewards))
            print(x.shape)
            print(self.K + 3)
            x = np.vstack((np.zeros((self.N - observations.shape[0], x.shape[1])), x)) # pad x with 0s
        else: # no actions and rewards yet
            x = np.zeros((self.N, self.K + 3))
            if observations.size > 0: # if we already observe the first state
                x[self.N-1, 1:2] = observations[0, :]

        x = torch.from_numpy(x).float()
        x = self.encoder(x) # result: traj_len x 32
        x = self.value_encoder(x) #result traj:len x 16
        x = x.unsqueeze(1)  # add a new dimension at the second dim
        x = self.tc_1(x)
        x = self.tc_2(x)
        x = self.attention_1(x)
        x = self.affine_2(x)
        x = x[self.N-1, :, :].squeeze()  # pick_last_action
        if (self.non_linearity):
            x = self.non_linearity(x)

        if keep:
            self.observations = observations
            self.actions = actions
            self.rewards = rewards

        return x.unsqueeze(0).unsqueeze(0)

    def reset_hidden_state(self):
        self.observations = np.array([])
        self.actions = np.array([])
        self.rewards = np.array([])