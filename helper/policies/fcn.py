import torch
import torch.nn as nn
import torch.nn.functional as F
from helper.policies.policy import Policy


class FCNPolicy(Policy):
    def __init__(self, output_size, input_size=1, hidden_size=128):
        super(FCNPolicy, self).__init__(input_size, output_size)

        self.affine_1 = nn.Linear(1, hidden_size)
        self.affine_2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.affine_1(x))
        action_scores = self.affine_2(x)
        return F.softmax(action_scores, dim=1)
