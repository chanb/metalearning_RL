class PPO:
  def __init__(self, model, optimizer, ppo_epochs, mini_batchsize, batchsize, clip_param, vf_coef, ent_coef):
    self.model = model
    self.optimizer = optimizer
    self.ppo_epochs = ppo_epochs
    self.batchsize = batchsize
    self.mini_batchsize = mini_batchsize
    self.clip_param = clip_param
    self.vf_coef = vf_coef
    self.ent_coef = ent_coef

  def update(self):
    pass

