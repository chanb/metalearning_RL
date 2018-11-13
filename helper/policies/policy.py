import torch
import torch.nn as nn


def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()
    elif isinstance(module, nn.GRU):
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

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
