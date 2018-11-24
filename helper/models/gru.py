import torch
import torch.nn as nn
import math
from helper.values import GRUValue
from helper.policies import GRUPolicy
import torch.nn.functional as F
from torch.distributions import Normal
from helper.policies.policy import weight_init


class GRUActorCritic(nn.Module):
    def __init__(self, output_size, init_state, input_size=1, hidden_size=256, non_linearity='none', init_std=1.0, min_std=1e-6):
        super(GRUActorCritic, self).__init__()

        # self.critic = GRUValue(output_size, init_state, input_size, hidden_size, non_linearity=non_linearity)
        # self.actor = GRUPolicy(output_size, init_state, input_size, hidden_size)

        # self.is_recurrent = True
        self.is_recurrent = True
        self.hidden_size = hidden_size
        self.min_log_std = math.log(min_std)
        self.prev_state = init_state

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.relu1 = nn.ReLU()
        self.policy = nn.Linear(hidden_size, output_size)
        self.value = nn.Linear(hidden_size, 1)
        if non_linearity == 'sigmoid':
            self.non_linearity = nn.Sigmoid()
        elif non_linearity == 'tanh':
            self.non_linearity = nn.Tanh()
        elif non_linearity == 'relu':
            self.non_linearity = nn.ReLU()
        else:
            self.non_linearity = None

        self.sigma = nn.Parameter(torch.Tensor(output_size))
        self.sigma.data.fill_(math.log(init_std))
        self.apply(weight_init)

    def forward(self, x, keep=True):
        # val = self.critic(x, keep)
        # mu = self.actor(x, keep)
        # return mu, val
        x, h = self.gru(x, self.prev_state)
        if keep:
            self.prev_state = h
        x = self.relu1(x)
        val = self.value(x)
        if self.non_linearity:
            val = self.non_linearity(val)
        mu = self.policy(x).detach()
        scale = torch.exp(torch.clamp(self.sigma, min=self.min_log_std)).detach()

        return Normal(loc=mu, scale=scale), val.detach()

    def reset_hidden_state(self):
        # self.critic.reset_hidden_state()
        # self.actor.reset_hidden_state()
        self.prev_state = torch.randn(1, 1, 256)
