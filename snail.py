import gym
import numpy as np
import argparse

import torch
import torch.optim as optim

import helper.envs
from helper.policies import SNAILPolicy, LinearEmbedding
from helper.models import SNAILActorCritic
from helper.algo import ppo, reinforce
import os

parser = argparse.ArgumentParser(description='SNAIL for MAB, MDP and 2DNav')

parser.add_argument('--num_actions', type=int, default=1,
                    help='number of arms for MAB or number of actions for MDP (default: 1)')
parser.add_argument('--max-num-traj', type=int, default=20, help='maximum number of trajectories to run (default: 20)')

parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
parser.add_argument('--max-traj-len', type=int, default=100, help='maximum trajectory length (default: 100)')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.95, help='lambda in GAE (default: 0.95)')
parser.add_argument('--learning-rate', type=float, default=1e-2,
                    help='learning rate for gradient descent (default: 1e-2)')

parser.add_argument('--num-tasks', type=int, default=200, help='number of similar tasks to run (default: 200)')
parser.add_argument('--algo', type=str, default='ppo',
                    help='algorithm to use [reinforce/ppo] (default: ppo)')
parser.add_argument('--mini-batch-size', type=int, default=1,
                    help='minimum batch size (default: 5) - needs to be <= max_traj_len')
parser.add_argument('--ppo-epochs', type=int, default=1, help='ppo epoch (default: 1)')
parser.add_argument('--task', type=str, default='2dnav', help='the task to learn [bandit, mdp, 2dnav] (default: 2dnav)')

parser.add_argument('--clip-param', type=float, default=0.2, help='clipping parameter for PPO (default: 0.2)')
parser.add_argument('--non-linearity', help='non linearity function following last output layer')

args = parser.parse_args()

eps = np.finfo(np.float32).eps.item()
out_folder = './saves/snail'
out_model = '{}/{}_{}_{}_{}_adam_lr{}_numtasks{}.pt'.format(out_folder, args.algo, args.task, args.num_actions,
                                                            args.max_num_traj, args.learning_rate, args.num_tasks)


def meta_train():
    task = ''
    if args.task == 'bandit':
        task = "Bandit-K{}-v0".format(args.num_actions)
        num_actions = args.num_actions
        num_states = 1
        non_linearity = 'sigmoid'
    elif args.task == 'mdp':
        task = "TabularMDP-v0"
        num_actions = 5
        num_states = 10
        non_linearity = 'none'
    elif args.task == '2dnav':
        task = "2DNavigation-v0"
        num_actions = 1
    else:
        print('Invalid Task')
        return

    if args.non_linearity:
        non_linearity = args.non_linearity

    fcn = LinearEmbedding(input_size=2 + num_states + num_actions, output_size=32)
    if args.algo == 'reinforce':
        policy = SNAILPolicy(num_actions, args.max_num_traj, args.max_traj_len, fcn)
        optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate)
        _, _, _, model = reinforce(policy, optimizer, task, num_actions, args.num_tasks, args.max_num_traj,
                                   args.max_traj_len,
                                   args.gamma)
    elif args.algo == 'ppo':
        model = SNAILActorCritic(num_actions, args.max_num_traj, args.max_traj_len, fcn, non_linearity=non_linearity)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
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
