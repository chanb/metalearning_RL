import torch
import torch.nn as nn


class Value(nn.Module):
    def __init__(self, output_size):
        super(Value, self).__init__()
        self.is_recurrent = False
        self.saved_log_probs = []
        self.output_size = output_size

    def forward(self, x):
        return 1
