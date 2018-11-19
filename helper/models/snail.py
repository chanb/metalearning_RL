import torch
import torch.nn as nn
from helper.policies import SNAILPolicy
from helper.values import SNAILValue
import torch.nn.functional as F
from torch.distributions import Categorical


class SNAILActorCritic(nn.Module):
    def __init__(self, output_size, max_num_traj, max_traj_len, encoder, input_size=1, policy_hidden_size=32, value_hidden_size=16, non_linearity='none'):
        super(SNAILActorCritic, self).__init__()
        self.is_recurrent = True
        self.critic = SNAILValue(output_size, max_num_traj, max_traj_len, encoder,
                                 encoder_hidden_size=policy_hidden_size, hidden_size=value_hidden_size,
                                 non_linearity=non_linearity)
        self.actor = SNAILPolicy(output_size, max_num_traj, max_traj_len, encoder, hidden_size=policy_hidden_size)

    def forward(self, x, keep=True):
        if self.past.size()[0] == 0:
            x = x
        elif self.past.shape[0] >= self.T:
            x = torch.cat((self.past[1:self.T, :, :], x))
        else:
            x = torch.cat((self.past, x))
        # if keep and not_zero > 0:
        if keep:
            self.past = x
        if x.shape[0] < self.T:
            x = torch.cat((torch.FloatTensor(self.T - x.shape[0], x.shape[1], x.shape[2]).zero_(), x))
        x = self.encoder(x)
        val = self.critic(x, keep)
        mu = self.actor(x, keep)
        return mu, val

    def reset_hidden_state(self):
        # self.critic.reset_hidden_state()
        # self.actor.reset_hidden_state()
        self.past = torch.FloatTensor()