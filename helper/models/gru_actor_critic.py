import torch
import torch.nn as nn
from helper.values import GRUValue
from helper.policies import GRUPolicy
import torch.nn.functional as F
from torch.distributions import Categorical


class GRUActorCritic(nn.Module):
    def __init__(self, output_size, init_state, input_size=1, hidden_size=256, std=0.0):
        super(GRUActorCritic, self).__init__()

        self.critic = GRUValue(output_size, init_state, input_size, hidden_size)
        self.actor = GRUPolicy(output_size, init_state, input_size, hidden_size)

        self.log_std = nn.Parameter(torch.ones(1, output_size) * std)
        self.is_recurrent = True

    def forward(self, x):
        val = self.critic(x)
        mu = self.actor(x)
        return mu, val

    def reset_hidden_state(self):
        self.critic.reset_hidden_state()
        self.actor.reset_hidden_state()
