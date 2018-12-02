import numpy as np
import torch

CLIP_BASE = 1.0

# This performs PPO update using the Sampler storage
class PPO:
  def __init__(self, model, optimizer, ppo_epochs, mini_batchsize, batchsize, clip_param, vf_coef, ent_coef, max_grad_norm, target_kl):
    self.model = model
    self.optimizer = optimizer
    self.ppo_epochs = ppo_epochs
    self.batchsize = batchsize
    self.mini_batchsize = mini_batchsize
    self.clip_param = clip_param
    self.vf_coef = vf_coef
    self.ent_coef = ent_coef
    self.max_grad_norm = max_grad_norm
    self.target_kl = target_kl

  # Samples minibatch
  def ppo_iter(self, mini_batch_size, states, actions, log_probs, returns, advantages, values, hidden_states):
    batch_size = states.size(0)
    # print('log_prob: {} \nstate: {}\naction: {} \nreward: {} \nadv:{}, \nvalue: {}'.format(log_probs.shape, states.shape, actions.shape, returns.shape, advantages.shape, values.shape, hidden_states.shape))
    for _ in range(batch_size // mini_batch_size):
      rand_ids = np.random.choice(batch_size, mini_batch_size, True)
      yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantages[rand_ids, :], values[rand_ids, :], hidden_states[rand_ids, :]

  # Perform PPO Update
  def update(self, sampler):
    print('PPO Update')
    for epoch in range(self.ppo_epochs):
      print('PPO epoch {}'.format(epoch))
      for state, action, old_log_probs, ret, advantage, old_value, hidden_state in self.ppo_iter(self.mini_batchsize, sampler.states, sampler.actions, sampler.log_probs, sampler.returns, sampler.advantages, sampler.values, sampler.get_hidden_state()):
        # Computes the new log probability from the updated model
        new_log_probs = []
        values = []
        for sample in range(self.mini_batchsize):
          dist, value, _, = self.model(state[sample].unsqueeze(0), hidden_state[sample].unsqueeze(0), to_print=False)
          entropy = dist.entropy().mean()
          new_log_probs.append(dist.log_prob(action[sample]).unsqueeze(0))
          values.append(value)
          # print('value: {} \nentropy: {} \nlog_prob: {}'.format(value, entropy, dist.log_prob(action[sample])))
        new_log_probs = torch.cat(new_log_probs)
        values = torch.cat(values)
        
        # Compute the values for objective function 
        # (ratio for some reason favours bad action. Probably the reason why it's not converging with negated obj func)
        ratio = torch.exp(new_log_probs - old_log_probs).unsqueeze(2)
        kl = (old_log_probs - new_log_probs).mean()
        if kl > 1.5 * self.target_kl:
          print('Early breaking due to high KL')
          break
        
        surr_1 = ratio * advantage
        surr_2 = torch.clamp(ratio, CLIP_BASE - self.clip_param, CLIP_BASE + self.clip_param) * advantage
        
        # Clipped Surrogate Objective Loss
        actor_loss = torch.min(surr_1, surr_2).mean()
        
        value_clipped = old_value + torch.clamp(old_value - values, -self.clip_param, self.clip_param)
        val_1 = (ret - values).pow(2)
        val_2 = (ret - value_clipped).pow(2)

        # Mean Squared Error Loss Function
        critic_loss = 0.5 * torch.min(val_1, val_2).pow(2).mean()
        
        # This is L(Clip) - c_1L(VF) + c_2L(S)
        # Take negative because we're doing gradient descent
        loss = -(actor_loss - self.vf_coef * critic_loss + self.ent_coef * entropy)
        # print('kl: {} \nactions: {} \nratio: {} \nadv: {} \nsurr1: {} \nsurr2: {} \nactor: {} \ncritic: {} \nloss: {}'.format(
        #   kl, action.squeeze(1).squeeze(1), ratio, advantage, surr_1, surr_2, actor_loss, critic_loss, loss))

        self.optimizer.zero_grad()
        loss.backward()
        # Try clipping
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()


