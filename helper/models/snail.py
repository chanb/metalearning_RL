import torch
import torch.nn as nn
from helper.policies import SNAILPolicy
from helper.values import SNAILValue


class SNAILActorCritic(nn.Module):
  def __init__(self, output_size, max_num_traj, max_traj_len, encoder, input_size=1, policy_hidden_size=32, value_hidden_size=16, non_linearity='none'):
    super(SNAILActorCritic, self).__init__()
    self.K = output_size
    self.N = max_num_traj
    self.T = max_num_traj * max_traj_len
    self.is_recurrent = True
    self.critic = SNAILValue(output_size, max_num_traj, max_traj_len, encoder,
                              encoder_hidden_size=policy_hidden_size, hidden_size=value_hidden_size,
                              non_linearity=non_linearity)
    self.actor = SNAILPolicy(output_size, max_num_traj, max_traj_len, encoder, hidden_size=policy_hidden_size)

  def forward(self, x, hidden_state, to_print=True):
    val, critic_hidden_state = self.critic(x, hidden_state)
    dist, actor_hidden_state = self.actor(x, hidden_state, to_print)

    assert torch.all(torch.eq(critic_hidden_state, actor_hidden_state)), 'They should have same hidden state'

    return dist, val.unsqueeze(0), actor_hidden_state

  def init_hidden_state(self, batchsize=1):
    return torch.zeros([1, batchsize, self.T])