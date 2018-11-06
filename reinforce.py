import gym
import numpy as np
import argparse

import torch
import torch.optim as optim
from torch.distributions import Categorical

from helper.policies import FCNPolicy

parser = argparse.ArgumentParser(description='PyTorch REINFORCE Multi-armed Bandit')

parser.add_argument('--max_traj', type=int, default=10, help='maximum number of trajectories to run (default: 10)')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
parser.add_argument('--max_iter', type=int, default=1, help='maximum trajectory length (default: 1)')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor (default: 0.99)')
parser.add_argument('--learning_rate', type=float, default=1e-2, help='learning rate for gradient descent (default: 1e-2)')

args = parser.parse_args()

# Create environment and initialize seed
env = gym.make("Bandit-K10-v0")
env.seed(args.seed)
torch.manual_seed(args.seed)
eps = np.finfo(np.float32).eps.item()

def select_action(policy, state):
  state = torch.from_numpy(state).float().unsqueeze(0)
  probs = policy(state)
  m = Categorical(probs)
  print(m.probs)
  action = m.sample()
  policy.saved_log_probs.append(m.log_prob(action))
  return action.item()


def main():
  policy = FCNPolicy(10)
  optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate)
  
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
    policy_loss.backward()
    optimizer.step()
    del policy.saved_log_probs[:]


    print(actions)
    print(rewards)
    
    print('Episode {}\tLast length: {:5d}'.format(traj, horizon))

if __name__=='__main__':
  main()