import torch
import torch.nn as nn
from helper.policies import SNAILPolicy
from helper.values import SNAILValue


class SNAILActorCritic(nn.Module):
  def __init__(self, output_size, max_num_traj, max_traj_len, encoder, input_size=1, policy_hidden_size=32, value_hidden_size=16, non_linearity='none'):
    super(SNAILActorCritic, self).__init__()
    self.is_recurrent = True
    self.critic = SNAILValue(output_size, max_num_traj, max_traj_len, encoder,
                              encoder_hidden_size=policy_hidden_size, hidden_size=value_hidden_size,
                              non_linearity=non_linearity)
    self.actor = SNAILPolicy(output_size, max_num_traj, max_traj_len, encoder, hidden_size=policy_hidden_size)

  def forward(self, x, hidden_state):
    val, critic_hidden_state = self.critic(x, hidden_state)
    mu, actor_hidden_state = self.actor(x, hidden_state)
    return mu, val, actor_hidden_state, critic_hidden_state

  def init_hidden(self):
    return torch.FloatTensor()