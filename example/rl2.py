import envs
import gym
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from policy import FCN_Policy, GRU_Policy
from model import GRU_ActorCritic

parser = argparse.ArgumentParser(description='PyTorch REINFORCE Multi-armed Bandit')

parser.add_argument('--num_arms', type=int, default=5, help='number of arms for MAB (default: 5)')
parser.add_argument('--max_traj', type=int, default=10, help='maximum number of trajectories to run (default: 10)')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
parser.add_argument('--max_iter', type=int, default=1, help='maximum trajectory length (default: 1)')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor (default: 0.99)')
parser.add_argument('--learning_rate', type=float, default=1e-2, help='learning rate for gradient descent (default: 1e-2)')
parser.add_argument('--max_task', type=int, default=5, help='number of similar tasks to run (default: 5)')
parser.add_argument('--algo', type=str, default='reinforce', help='algorithm to use [reinforce/ppo] (default: reinforce)')
parser.add_argument('--mini_batch_size', type=int, default=1, help='minimum batch size (default: 5) - needs to be <= max_iter')
parser.add_argument('--ppo_epochs', type=int, default=1, help='ppo epoch (default: 1)')


args = parser.parse_args()

# Create environment and initialize seed
# env.seed(args.seed)
# torch.manual_seed(args.seed)
eps = np.finfo(np.float32).eps.item()

def select_action(policy, state):
  state = torch.from_numpy(state).float().unsqueeze(0)

  if policy.is_recurrent:
    state = state.unsqueeze(0)

  probs = policy(state)
  m = Categorical(probs)
  # print(m.probs)
  action = m.sample()
  # print(action)
  policy.saved_log_probs.append(m.log_prob(action))
  return action.item()

def reinforce():
  # TODO: Add randomize number of trajectories to run
  policy = GRU_Policy(args.num_arms, torch.randn(1, 1, 256))
  optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate)

  # Meta-Learning
  for task in range(args.max_task):
    print("Task {} ==========================================================================================================".format(task))
    env = gym.make("Bandit-K{}-v0".format(args.num_arms))

    # REINFORCE
    for traj in range(args.max_traj):
      state = env.reset()

      rewards = []
      actions = []
      for horizon in range(args.max_iter):
        action = select_action(policy, state)
        state, reward, done, info = env.step(action)
        
        actions.append(action)
        rewards.append(reward)
        if (done):
          break
      
      # Batch gradient descent
      R = 0
      policy_loss = []
      discounted_rewards = []
      traj_len = len(discounted_rewards)
      
      for r in rewards[::-1]:
        R = r + args.gamma * R
        discounted_rewards.insert(0, R)
      
      discounted_rewards = torch.tensor(discounted_rewards)
      
      if (traj_len > 1):
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + eps)

      # Compute loss and take gradient step
      for log_prob, reward in zip(policy.saved_log_probs, discounted_rewards):
        policy_loss.append(-log_prob * reward)
      optimizer.zero_grad()
      policy_loss = torch.cat(policy_loss).sum()
      policy_loss.backward(retain_graph=policy.is_recurrent)
      optimizer.step()
      del policy.saved_log_probs[:]

      print(actions)
      print(rewards)
      
      print('Episode {}\tLast length: {:5d}\tTask: {}'.format(traj, horizon, task))

    if policy.is_recurrent:
      policy.reset_hidden_state()

def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
  values = values + [next_value]
  gae = 0
  returns = []
  for step in reversed(range(len(rewards))):
    delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
    gae = delta + gamma * tau * masks[step] * gae
    returns.insert(0, gae + values[step])
  return returns

def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
  batch_size = states.size(0)
  for _ in range(batch_size // mini_batch_size):
    rand_ids = np.random.randint(0, batch_size, mini_batch_size)
    # print(rand_ids)
    # print(states)
    # print(actions)
    # print(log_probs)
    # print(returns)
    # print(advantages)
    yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantages[rand_ids, :]

def ppo_update(model, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
  for i in range(ppo_epochs):
    for state, action, log_prob, ret, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
      dist, value = model(state)

      entropy = dist.entropy().mean()
      new_log_probs = dist.log_prob(action)

      ratio = (new_log_probs - log_probs).exp()
      surr_1 = ratio * advantage
      surr_2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

      actor_loss = torch.min(surr_1, surr_2).mean()
      critic_loss = (ret - value).pow(2).mean()
      
      loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

      optimizer.zero_grad()
      loss.backward(retain_graph=model.is_recurrent)
      optimizer.step()

# Attempt to modify policy so it doesn't go too far
def ppo():
  model = GRU_ActorCritic(args.num_arms, torch.randn(1, 1, 256))
  optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

  # Meta-Learning
  for task in range(args.max_task):
    print("Task {} ==========================================================================================================".format(task))
    env = gym.make("Bandit-K{}-v0".format(args.num_arms))

    # PPO (Using actor critic style)
    for _ in range(args.max_traj):
      state = env.reset()

      log_probs = []
      values = []
      states = []
      actions = []
      rewards = []
      masks = []
      entropy = 0

      for _ in range(args.max_iter):
        state = torch.from_numpy(state).float().unsqueeze(0)

        if model.is_recurrent:
          state = state.unsqueeze(0)

        states.append(state)

        dist, value = model(state)
        print(dist.probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        state, reward, done, _ = env.step(action.item())

        entropy += dist.entropy().mean()

        log_probs.append(log_prob.unsqueeze(0).unsqueeze(0))
        actions.append(action.unsqueeze(0).unsqueeze(0))
        values.append(value)
        rewards.append(reward)
        masks.append(1 - done)

        if (done):
          break

      state = torch.from_numpy(state).float().unsqueeze(0)
      if model.is_recurrent:
        state = state.unsqueeze(0)

      _, next_val = model(state)
      returns = compute_gae(next_val, rewards, masks, values)
      
      # print(values)
      # print(returns)
      # print(log_probs)
      # print(states)
      
      returns = torch.cat(returns)
      values = torch.cat(values)
      log_probs = torch.cat(log_probs)
      states = torch.cat(states)
      actions = torch.cat(actions)

      advantage = returns - values

      print("DATA =====================")
      # print(returns)
      # print(values)
      # print(advantage)
      print(actions)
      # print(states)
      # print(log_probs)
      print(rewards)
      
      # This is where we compute loss and update the model
      ppo_update(model, optimizer, args.ppo_epochs, args.mini_batch_size, states, actions, log_probs, returns, advantage)

    if model.is_recurrent:
      model.reset_hidden_state()

def main():
  if args.algo == 'reinforce':
    reinforce()
  elif args.algo == 'ppo':
    ppo()
  else:
    print('Invalid learning algorithm')

  
if __name__=='__main__':
  main()