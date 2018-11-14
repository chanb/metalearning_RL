import torch
import torch.nn as nn
from helper.values.value import Value
import torch.nn.init as I
from helper.policies.policy import weight_init


class GRUValue(Value):
    def __init__(self, output_size, init_state, input_size=1, hidden_size=256):
        super(GRUValue, self).__init__(output_size)
        self.is_recurrent = True
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.relu1 = nn.ReLU()
        self.value = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.prev_state = init_state
        self.apply(weight_init)

    def forward(self, x):
        x, h = self.gru(x, self.prev_state)
        self.prev_state = h
        x = self.relu1(x)
        return self.sigmoid(self.value(x))

    def reset_hidden_state(self):
        self.prev_state = torch.randn(1, 1, 256)
