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
      policy_loss.backward(retain_graph=True)
      optimizer.step()
      del policy.saved_log_probs[:]

      print(actions)
      print(rewards)
      
      print('Episode {}\tLast length: {:5d}\tTask: {}'.format(traj, horizon, task))

    if policy.is_recurrent:
      policy.reset_hidden_state()

def ppo():
  model = GRU_ActorCritic(args.num_arms, torch.randn(1, 1, 256))
  optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

  # Meta-Learning
  for task in range(args.max_task):
    print("Task {} ==========================================================================================================".format(task))
    env = gym.make("Bandit-K{}-v0".format(args.num_arms))

    # PPO
    for traj in range(args.max_traj):
      

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