import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
from helper.policies.policy import Policy, weight_init


class GRUPolicy(Policy):
    def __init__(self, output_size, init_state, input_size=1, hidden_size=256):
        super(GRUPolicy, self).__init__(input_size, output_size)
        self.is_recurrent = True
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.relu1 = nn.ReLU()
        self.affine = nn.Linear(hidden_size, output_size)

        self.prev_state = init_state
        self.apply(weight_init)

    def forward(self, x, keep=True):
        x, h = self.gru(x, self.prev_state)
        if keep:
            self.prev_state = h
        x = self.relu1(x)
        x = self.affine(x).squeeze(0)
        return F.softmax(x, dim=1)

    def reset_hidden_state(self):
        self.prev_state = torch.randn(1, 1, 256)
