import gym
import numpy as np
import argparse
import pickle

import torch
import torch.optim as optim
from torch.distributions import Categorical

import helper.envs
from helper.policies import LinearEmbedding
from helper.models import GRUActorCritic, SNAILActorCritic, FCNActorCritic
from helper.metalearn import MetaLearner
from helper.sampler import Sampler
from helper.algo import PPO
import os

parser = argparse.ArgumentParser(description='RL2 for MAB and MDP')

parser.add_argument('--metalearn_epochs', type=int, default=1000, help='number of epochs for meta learning')
parser.add_argument('--task', type=str, default='bandit', help='the task to learn [bandit, mdp] (default: bandit)')
parser.add_argument('--non_linearity', help='non linearity function following last output layer')
parser.add_argument('--learning_rate', type=float, default=3e-4, help='learning rate for optimizer (default: 3e-4)')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor (default: 0.99)')

parser.add_argument('--num_actions', type=int, default=5, help='number of arms for MAB or number of actions for MDP (default: 5)')
parser.add_argument('--num_tasks', type=int, default=1000, help='number of similar tasks to run (default: 100)')
parser.add_argument('--num_traj', type=int, default=10, help='number of trajectories to interact with (default: 10)')
parser.add_argument('--traj_len', type=int, default=1, help='fixed trajectory length (default: 1)')

parser.add_argument('--tau', type=float, default=0.95, help='GAE parameter (default: 0.95)')
parser.add_argument('--mini_batch_size', type=int, default=256, help='minibatch size for ppo update (default: 256)')
parser.add_argument('--batch_size', type=int, default=10000, help='batch size (default: 10000)')
parser.add_argument('--ppo_epochs', type=int, default=5, help='ppo epoch (default: 5)')
parser.add_argument('--clip_param', type=float, default=0.1, help='clipping parameter for PPO (default: 0.1)')

parser.add_argument('--vf_coef', type=float, default=0.5, help='value loss coefficient (default: 0.5)')
parser.add_argument('--ent_coef', type=float, default=0.01, help='entropy coefficient (default: 0.01)')
parser.add_argument('--max_grad_norm', type=float, default=0.5, help='max norm of gradients (default: 0.5)')
parser.add_argument('--target_kl', type=float, default=0.01, help='max target kl (default: 0.01)')

parser.add_argument('--out_file', type=str, help='the output file that stores the model')

args = parser.parse_args()
tmp_folder = './tmp/'
eps = np.finfo(np.float32).eps.item()

# Performs meta training
def meta_train(metalearn_epochs, task, num_actions, num_states, num_tasks, num_traj, traj_len, ppo_epochs, mini_batchsize, batchsize, gamma, 
  tau, clip_param, learning_rate, vf_coef, ent_coef, max_grad_norm, target_kl, non_linearity, out_file):

  # Create the model
  model = GRUActorCritic(num_actions, 2 + num_states + num_actions, non_linearity=non_linearity)

  # model = FCNActorCritic(num_actions, num_states, non_linearity=non_linearity)
  # fcn = LinearEmbedding(input_size=2 + num_states + num_actions, output_size=32)
  # model = SNAILActorCritic(num_actions, args.num_traj, args.traj_len, fcn, non_linearity=non_linearity)

  # Set the optimizer
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  
  # Create the agent that uses PPO
  agent = PPO(model, optimizer, ppo_epochs, mini_batchsize, batchsize, clip_param, vf_coef, ent_coef, max_grad_norm, target_kl)
  meta_learner = MetaLearner(task, num_actions, num_states, num_tasks, num_traj, traj_len)

  # meta_learner.tasks[0] = {'mean': [1,0,0,0,0]}
  # meta_learner.tasks[1] = {'mean': [0,1,0,0,0]}
  # meta_learner.tasks[2] = {'mean': [0,0,1,0,0]}

  for i in range(metalearn_epochs):
    print('Meta-train epoch {}'.format(i))

    # Decay learning rate
    decayed_lr = learning_rate - (learning_rate * (i / float(metalearn_epochs)))
    for param_group in agent.optimizer.param_groups:
      param_group['lr'] = decayed_lr

    # Linear decay on clipping parameter
    agent.clip_param = clip_param  * (1 - i / float(metalearn_epochs))

    meta_learner.train(model, agent, gamma, tau)

    # Temporary save model
    temp_out_file = '{}{}_{}'.format(tmp_folder, i, out_file)
    if os.path.exists(temp_out_file):
      os.remove(temp_out_file)
    torch.save(model, temp_out_file)

  return model

def main():
  assert (args.task == 'bandit' or args.task == 'mdp'), 'Invalid Task'
  assert (args.mini_batch_size <= args.batch_size), 'Mini-batch size needs to be <= batch size'
  task = ''
  if args.task == 'bandit':
    task = "Bandit-K{}-v0".format(args.num_actions)
    num_actions = args.num_actions
    num_states = 1
  elif args.task == 'mdp':
    task = "TabularMDP-v0"
    num_actions = 5
    num_states = 10

  if (not os.path.exists(tmp_folder)):
    os.mkdir(tmp_folder)

  model = meta_train(args.metalearn_epochs, task, num_actions, num_states, args.num_tasks, args.num_traj, args.traj_len, args.ppo_epochs, 
    args.mini_batch_size, args.batch_size, args.gamma, args.tau, args.clip_param, args.learning_rate, args.vf_coef, 
    args.ent_coef, args.max_grad_norm, args.target_kl, args.non_linearity, args.out_file)

  if (model):
    if os.path.exists(args.out_file):
      os.remove(args.out_file)
    torch.save(model, args.out_file)
  else:
    print('Model is not generated')


if __name__ == '__main__':
  main()