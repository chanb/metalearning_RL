import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from helper.policies.policy import weight_init


class FCNActorCritic(nn.Module):
    def __init__(self, output_size, input_size=1, hidden_size=128, non_linearity='none'):
        super(FCNActorCritic, self).__init__()
        self.is_recurrent = False
        self.affine_1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.actor = nn.Linear(hidden_size, output_size)
        self.critic = nn.Linear(hidden_size, 1)
        if non_linearity == 'sigmoid':
            self.non_linearity = nn.Sigmoid()
        elif non_linearity == 'tanh':
            self.non_linearity = nn.Tanh()
        elif non_linearity == 'relu':
            self.non_linearity = nn.ReLU()
        else:
            self.non_linearity = None
        self.apply(weight_init)

    def forward(self, x, keep=True):
        x = self.affine_1(x)
        x = self.relu(x)
        val = self.critic(x)
        if (self.non_linearity):
            val = self.non_linearity(val)
        dist = self.actor(x).squeeze(0)
        return dist, val