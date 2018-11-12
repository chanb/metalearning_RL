import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
from helper.policies.policy import Policy


class GRUPolicy(Policy):
    def __init__(self, output_size, init_state, input_size=1, hidden_size=256):
        super(GRUPolicy, self).__init__(input_size, output_size)
        self.is_recurrent = True
        self.hidden_size = hidden_size
        self.init_state = init_state

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        
                
        self.affine = nn.Linear(hidden_size, output_size)
        
        I.xavier_normal(self.affine.weight)

        self.prev_state = self.init_state

    def forward(self, x):
        x, h = self.gru(x, self.prev_state)
        self.prev_state = h
        x = self.affine(x).squeeze(0)
        return F.softmax(x, dim=1)

    def reset_hidden_state(self):
        #self.gru.reset_parameters()
        self.prev_state = self.init_state
