import numpy as np
import torch

clip_base = 1.0
class PPO:
  def __init__(self, model, optimizer, ppo_epochs, mini_batchsize, batchsize, clip_param, vf_coef, ent_coef, max_grad_norm):
    self.model = model
    self.optimizer = optimizer
    self.ppo_epochs = ppo_epochs
    self.batchsize = batchsize
    self.mini_batchsize = mini_batchsize
    self.clip_param = clip_param
    self.vf_coef = vf_coef
    self.ent_coef = ent_coef
    self.max_grad_norm = max_grad_norm

  def ppo_iter(self, mini_batch_size, states, actions, log_probs, returns, advantages, hidden_states):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
      rand_ids = np.random.choice(batch_size, mini_batch_size, False)
      yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], \
            advantages[rand_ids, :], hidden_states[rand_ids, :]

  def update(self, sampler):
    for epoch in range(self.ppo_epochs):
      for state, action, old_log_probs, ret, advantage, hidden_state in self.ppo_iter(self.mini_batchsize, sampler.states, sampler.actions, sampler.log_probs, sampler.returns,
        sampler.advantages, sampler.hidden_states):
        # Computes the new log probability from the updated model
        new_log_probs = []
        values = []
        for sample in range(self.mini_batchsize):
          dist, value, _, = self.model(state[sample].unsqueeze(0), hidden_state[sample].unsqueeze(0), to_print=False)
          entropy = dist.entropy().mean()
          new_log_probs.append([dist.log_prob(action[sample])])
          values.append(value)
        new_log_probs = torch.tensor(new_log_probs)
        values = torch.tensor(values)

        # Compute the values for objective function 
        # (ratio for some reason favours bad action. Probably the reason why it's not converging with negated obj func)
        ratio = torch.exp(new_log_probs - old_log_probs.squeeze(1))

        surr_1 = ratio * advantage
        surr_2 = torch.clamp(ratio, clip_base - self.clip_param, clip_base + self.clip_param) * advantage

        # Clipped Surrogate Objective Loss
        actor_loss = torch.min(surr_1, surr_2).mean()

        # Mean Squared Error Loss Function
        critic_loss = (ret - values).pow(2).mean()

        # This is L(Clip) - c_1L(VF) + c_2L(S)
        # Take negative because we're doing gradient descent
        loss = -(actor_loss - self.vf_coef * critic_loss + self.ent_coef * entropy)

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        # Try clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()


