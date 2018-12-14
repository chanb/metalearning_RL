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


class LinearEmbedding(nn.Module):
  def __init__(self, input_size=1, output_size=32):
    super(LinearEmbedding, self).__init__()
    self.fcn = nn.Linear(input_size, output_size)

  def forward(self, x):
    return self.fcn(x)


class CausalConv1D(nn.Module):
  def __init__(self, input_size, dilation_rate, output_size, kernel_size=2, stride=1, groups=1, bias=True):
    super(CausalConv1D, self).__init__()
    self.dilation_rate = dilation_rate

    padding = max(0, ((output_size - 1) * stride + kernel_size - input_size)//2)
    print(padding)
    self.conv1d = nn.Conv1d(input_size, output_size, kernel_size, stride, padding, dilation_rate, groups, bias)

  def forward(self, x):
    # Input shape: (batch_size N, input_size C, sequence_length T)
    # Output shape: (batch_size N, output_size D, sequence length T)
    return self.conv1d(x)[:, :, :-self.dilation_rate]