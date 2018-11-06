import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from helper.policies.policy import Policy, weight_init
from collections import OrderedDict

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