import torch
import torch.nn as nn


def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()

     #   nn.init.orthogonal_(module.weight, 1)
     #   module.bias.data.zero_()


# Generic policy
class Policy(nn.Module):
    def __init__(self, input_size, output_size):
        super(Policy, self).__init__()
        self.is_recurrent = False
        self.saved_log_probs = []
        self.output_size = output_size
        self.input_size = input_size

    #def forward(self, x):
    #    return torch.tensor(1 / self.output_size).repeat(self.output_size)
