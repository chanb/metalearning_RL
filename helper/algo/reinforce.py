import numpy as np
from helper.algo.algorithm import PolicyGradientAlgorithm

class REINFORCE(PolicyGradientAlgorithm):
  def __init__(self, model, optimizer, baseline):
    super(REINFORCE, self).__init__(model, optimizer)
    self.baseline = baseline

  def update(self):
    pass