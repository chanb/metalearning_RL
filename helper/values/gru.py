import torch
import torch.nn as nn
from helper.values.value import Value
import torch.nn.init as I
from helper.policies.policy import weight_init


class GRUValue(Value):
    def __init__(self, output_size, init_state, input_size=1, hidden_size=256, non_linearity='none'):
        super(GRUValue, self).__init__(output_size)
        self.is_recurrent = True
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.relu1 = nn.ReLU()
        self.value = nn.Linear(hidden_size, 1)
        if non_linearity == 'sigmoid':
            self.non_linearity = nn.Sigmoid()
        elif non_linearity == 'tanh':
            self.non_linearity = nn.Tanh()
        elif non_linearity == 'relu':
            self.non_linearity = nn.ReLU()
        else:
            self.non_linearity = None
        self.prev_state = init_state
        self.apply(weight_init)

    def forward(self, x):
        x, h = self.gru(x, self.prev_state)
        self.prev_state = h
        x = self.relu1(x)
        x = self.value(x)
        if (self.non_linearity):
            x = self.non_linearity(x)
        return x

    def reset_hidden_state(self):
        self.prev_state = torch.randn(1, 1, 256)
