import gym
import numpy as np
import argparse
import pickle

import torch
import torch.optim as optim
from torch.distributions import Categorical

import helper.envs
from helper.policies import GRUPolicy, FCNPolicy
from helper.models import GRUActorCritic
from helper.algo import ppo, reinforce
import os

parser = argparse.ArgumentParser(description='RL2 for MAB and MDP')

parser.add_argument('--num_actions', type=int, default=5,
                    help='number of arms for MAB or number of actions for MDP (default: 5)')
parser.add_argument('--max_num_traj', type=int, default=10, help='maximum number of trajectories to run (default: 10)')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
parser.add_argument('--max_traj_len', type=int, default=1, help='maximum trajectory length (default: 1)')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.95, help='lambda in GAE (default: 0.95)')
parser.add_argument('--learning_rate', type=float, default=1e-2,
                    help='learning rate for gradient descent (default: 1e-2)')
parser.add_argument('--num_tasks', type=int, default=5, help='number of similar tasks to run (default: 5)')
parser.add_argument('--algo', type=str, default='reinforce',
                    help='algorithm to use [reinforce/ppo] (default: reinforce)')
parser.add_argument('--mini_batch_size', type=int, default=1,
                    help='minimum batch size (default: 5) - needs to be <= max_traj_len')
parser.add_argument('--ppo_epochs', type=int, default=1, help='ppo epoch (default: 1)')
parser.add_argument('--task', type=str, default='bandit', help='the task to learn [bandit, mdp] (default: bandit)')

parser.add_argument('--max_num_traj_eval', type=int, default=1000, help='maximum number of trajectories during evaluation (default: 1000)')
parser.add_argument('--clip_param', type=float, default=0.2, help='clipping parameter for PPO (default: 0.2)')
parser.add_argument('--non_linearity', help='non linearity function following last output layer')

args = parser.parse_args()

eps = np.finfo(np.float32).eps.item()
out_folder = './saves/rl2'
out_model = '{}/{}_{}_{}_{}_adam_lr{}_numtasks{}.pt'.format(out_folder, args.algo, args.task, args.num_actions,
                                                            args.max_num_traj, args.learning_rate, args.num_tasks)
result_folder = './logs/rl2'
out_result = '{}/{}_{}_{}_{}_adam_lr{}_numtasks{}.pkl'.format(result_folder, args.algo, args.task, args.num_actions,
                                                              args.max_num_traj, args.learning_rate, args.num_tasks)

def meta_train():
    task = ''
    if args.task == 'bandit':
        task = "Bandit-K{}-v0".format(args.num_actions)
        num_actions = args.num_actions
        num_states = 1
        non_linearity = 'tanh'
    elif args.task == 'mdp':
        task = "TabularMDP-v0"
        num_actions = 5
        num_states = 10
        non_linearity = 'none'
    else:
        print('Invalid Task')
        return

    if (args.non_linearity):
        non_linearity = args.non_linearity

    if args.algo == 'reinforce':
        # policy = FCNPolicy(num_actions, 1)
        policy = GRUPolicy(num_actions, torch.randn(1, 1, 256), input_size=2 + num_states + num_actions)
        optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate)
        _, _, _, model = reinforce(policy, optimizer, task, num_actions, args.num_tasks, args.max_num_traj, args.max_traj_len,
                  args.gamma)
    elif args.algo == 'ppo':
        # model = GRUActorCritic(num_actions, torch.randn(1, 1, 256), 4)
        model = GRUActorCritic(num_actions, torch.randn(1, 1, 256), 2 + num_states + num_actions, non_linearity=non_linearity)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        # optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
        _, _, _, model = ppo(model, optimizer, task, num_actions, args.num_tasks, args.max_num_traj, args.max_traj_len,
            args.ppo_epochs, args.mini_batch_size, args.gamma, args.tau, args.clip_param)
    else:
        print('Invalid learning algorithm')

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    if (model):
        if os.path.exists(out_model):
            os.remove(out_model)
        torch.save(model, out_model)

if __name__ == '__main__':
    print("TRAINING MODEL ========================================================================")
    meta_train()
