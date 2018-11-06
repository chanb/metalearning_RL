import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from collections import OrderedDict

from blocks import *


def weight_init(module):
  if isinstance(module, nn.Linear):
    nn.init.xavier_uniform_(module.weight)
    module.bias.data.zero_()

      
# Generic policy
class Policy(nn.Module):
  def __init__(self, input_size, output_size):
    super(Policy, self).__init__()
    self.is_recurrent = False
    self.saved_log_probs = []
    self.output_size = output_size
    self.input_size = input_size
  

  def forward(self, x):
    return torch.tensor(1/self.output_size).repeat(self.output_size)


# Fully Connected Network
class FCN_Policy(Policy):
  def __init__(self, output_size, input_size = 1):
    super(FCN_Policy, self).__init__(input_size, output_size)

    self.affine_1 = nn.Linear(1, 128)
    self.affine_2 = nn.Linear(128, output_size)


  def forward(self, x):
    x = F.relu(self.affine_1(x))
    action_scores = self.affine_2(x)
    return F.softmax(action_scores, dim=1)


# SNAIL
class SNAIL_Policy(Policy):
  # K arms, trajectory of length N
  def __init__(self, output_size, traj_len, input_size = 1):
    super(SNAIL_Policy, self).__init__(input_size, output_size)
    self.K = output_size
    self.N = traj_len

    num_filters = int(math.ceil(math.log(output_size * traj_len + 1)))
    num_channels = output_size + 1

    self.affine_1 = nn.Linear(num_channels, 32)
    num_channels += 32
    
    self.tc_1 = TCBlock(num_channels, traj_len, 32)
    num_channels += num_filters * 32

    self.tc_2 = TCBlock(num_channels, traj_len, 32)
    num_channels += num_filters * 32

    self.attention_1 = AttentionBlock(num_channels, 32, 32)
    num_channels += 32

    self.affine_2 = nn.Linear(num_channels, output_size)

  def forward(self, x):
    x = self.affine_1(x)
    x = self.tc_1(x)
    x = self.tc_2(x)
    x = self.attention_1(x)
    x = self.affine_2(x)
    return F.softmax(x, dim=1)


# GRU
class GRU_Policy(Policy):
  def __init__(self, output_size, init_state, input_size=1, hidden_size=256):
    super(GRU_Policy, self).__init__(input_size, output_size)
    self.is_recurrent = True
    self.hidden_size = hidden_size
    self.init_state = init_state

    self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
    self.affine = nn.Linear(hidden_size, output_size)

    self.prev_state = self.init_state


  def forward(self, x):
    x, h = self.gru(x, self.prev_state)
    self.prev_state = h
    x = self.affine(x).squeeze(0)
    return F.softmax(x, dim=1)


  def reset_hidden_state(self):
    self.prev_state = self.init_state


# Categorical MLP
class CategoricalMLPPolicy(Policy):
  """Policy network based on a multi-layer perceptron (MLP), with a
  `Categorical` distribution output. This policy network can be used on tasks
  with discrete action spaces (eg. `TabularMDPEnv`). The code is adapted from
  https://github.com/cbfinn/maml_rl2/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/sandbox/rocky/tf/policies/maml_minimal_categorical_mlp_policy.py
  """

  def __init__(self, input_size, output_size, hidden_sizes=(), nonlinearity=F.relu):
    super(CategoricalMLPPolicy, self).__init__(input_size, output_size)
    self.hidden_sizes = hidden_sizes
    self.nonlinearity = nonlinearity
    self.num_layers = len(hidden_sizes) + 1

    layer_sizes = (input_size,) + hidden_sizes + (output_size,)
    for i in range(1, self.num_layers + 1):
      self.add_module('layer{0}'.format(i),
                      nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
    self.apply(weight_init)


  def update_params(self, loss, step_size=0.5, first_order=False):
    """Apply one step of gradient descent on the loss function `loss`, with
    step-size `step_size`, and returns the updated parameters of the neural
    network.
    """
    grads = torch.autograd.grad(loss, self.parameters(),
                                create_graph=not first_order)
    updated_params = OrderedDict()
    for (name, param), grad in zip(self.named_parameters(), grads):
      updated_params[name] = param - step_size * grad

    return updated_params


  def forward(self, input, params=None):
    if params is None:
      params = OrderedDict(self.named_parameters())
    output = input
    for i in range(1, self.num_layers):
      output = F.linear(output,
                        weight=params['layer{0}.weight'.format(i)],
                        bias=params['layer{0}.bias'.format(i)])
      output = self.nonlinearity(output)
    logits = F.linear(output,
                      weight=params['layer{0}.weight'.format(self.num_layers)],
                      bias=params['layer{0}.bias'.format(self.num_layers)])

    return Categorical(logits=logits)