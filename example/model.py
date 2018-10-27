import torch
import torch.nn as nn
import value
import policy
from torch.distributions import Normal

class GRU_ActorCritic(nn.Module):
  def __init__(self, num_arms, init_state, hidden_size=256, std=0.0):
    super(GRU_ActorCritic, self).__init__()

    self.critic = value.GRU_Value(num_arms, init_state, hidden_size)
    self.actor = policy.GRU_Policy(num_arms, init_state, hidden_size)

    self.log_std = nn.Parameter(torch.ones(1, num_arms) * std)
    self.is_recurrent = True


  def forward(self, x):
    val = self.critic(x)
    mov = self.actor(x)
    std = self.log_std.exp().expand_as(mov)
    dist = Normal(mov, std)
    return dist, val


  def reset_hidden_state(self):
    self.critic.reset_hidden_state()
    self.actor.reset_hidden_state()