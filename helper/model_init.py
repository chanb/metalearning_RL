import torch
import torch.nn as nn

def weight_init(module):
  if isinstance(module, nn.Linear):
    nn.init.xavier_uniform_(module.weight)
    module.bias.data.zero_()
  elif isinstance(module, nn.GRU):
    for name, param in module.named_parameters():
      if 'weight_ih' in name:
        nn.init.xavier_uniform_(param)
      elif 'weight_hh' in name:
        nn.init.orthogonal_(param)
      elif 'bias' in name:
        nn.init.constant_(param, 0)