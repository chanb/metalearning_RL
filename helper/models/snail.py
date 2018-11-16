import torch
import torch.nn as nn
from helper.policies import SNAILPolicy
from helper.values import SNAILValue
import torch.nn.functional as F
from torch.distributions import Categorical


class SNAILActorCritic(nn.Module):
    def __init__(self, output_size, traj_len, encoder, input_size=1, policy_hidden_size=32, value_hidden_size=16):
        super(SNAILActorCritic, self).__init__()
        self.is_recurrent = True
        self.critic = SNAILValue(output_size=output_size, traj_len=traj_len, encoder=encoder,
                                 encoder_hidden_size=policy_hidden_size, hidden_size=value_hidden_size)
        self.actor = SNAILPolicy(output_size=output_size, traj_len=traj_len, encoder=encoder, hidden_size=policy_hidden_size)

    def forward(self, x, keep=True):
        val = self.critic(x, keep)
        mu = self.actor(x, keep)
        return mu, val

    def reset_hidden_state(self):
        self.critic.reset_hidden_state()
        self.actor.reset_hidden_state()
