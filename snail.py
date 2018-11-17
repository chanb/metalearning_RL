import gym
import numpy as np
import argparse
import pickle

import torch
import torch.optim as optim
from torch.distributions import Categorical

import helper.envs
from helper.policies import SNAILPolicy, LinearEmbedding
from helper.models import SNAILActorCritic
from helper.algo import ppo, reinforce
import os

parser = argparse.ArgumentParser(description='SNAIL for MAB and MDP')

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

parser.add_argument('--max_num_traj_eval', type=int, default=1000,
                    help='maximum number of trajectories during evaluation (default: 1000)')
parser.add_argument('--clip_param', type=float, default=0.2, help='clipping parameter for PPO (default: 0.2)')
parser.add_argument('--eval', type=int, default=1, help='do evaluation only (default: 1)')
parser.add_argument('--non_linearity', help='non linearity function following last output layer')

parser.add_argument('--eval_model', help='the model to evaluate')

args = parser.parse_args()

eps = np.finfo(np.float32).eps.item()
out_folder = './saves/snail'
out_model = '{}/{}_{}_{}_{}.pt'.format(out_folder, args.algo, args.task, args.num_actions, args.max_num_traj)
result_folder = './logs/snail'
out_result = '{}/{}_{}_{}_{}.pkl'.format(result_folder, args.algo, args.task, args.num_actions, args.max_num_traj)


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
    else:
        print('Invalid Task')
        return

    if (args.non_linearity):
        non_linearity = args.non_linearity

    fcn = LinearEmbedding(input_size=3+num_actions, output_size=32)
    if args.algo == 'reinforce':
        policy = SNAILPolicy(output_size=num_actions, traj_len=args.max_num_traj*args.max_traj_len, encoder=fcn)
        optimizer = optim.SGD(policy.parameters(), lr=args.learning_rate)
        _, _, _, model = reinforce(policy, optimizer, task, num_actions, args.num_tasks, args.max_num_traj,
                                   args.max_traj_len,
                                   args.gamma)
    elif args.algo == 'ppo':
        model = SNAILActorCritic(output_size=num_actions, traj_len=args.max_num_traj*args.max_traj_len, encoder=fcn, non_linearity=non_linearity)
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


def evaluate_model(eval_model):
    to_use = out_model
    if (eval_model):
        to_use = eval_model

    model = torch.load(to_use)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    task = ''
    if args.task == 'bandit':
        task = "Bandit-K{}-v0".format(args.num_actions)
        num_actions = args.num_actions
        num_states = 1
    elif args.task == 'mdp':
        task = "TabularMDP-v0"
        num_actions = 5
        num_states = 10
    else:
        print('Invalid Task')
        return

    if model.is_recurrent:
        model.reset_hidden_state()
    if args.algo == 'reinforce':
        all_rewards, all_states, all_actions, _ = reinforce(model, optimizer, task, num_actions, 1,
                                                            args.max_num_traj_eval, args.max_traj_len,
                                                            args.gamma)
    elif args.algo == 'ppo':
        all_rewards, all_states, all_actions, _ = ppo(model, optimizer, task, num_actions, 1, args.max_num_traj_eval,
                                                      args.max_traj_len,
                                                      args.ppo_epochs, args.mini_batch_size, args.gamma, args.tau,
                                                      args.clip_param, evaluate=True)
    else:
        print('Invalid learning algorithm')

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    with open(out_result, 'wb') as f:
        pickle.dump([all_rewards, all_actions, all_states, num_actions, num_states], f)

    idx = 0
    for traj in all_states[0]:
        idx += 1
        curr_traj = traj
        print('traj {} (length: {}) reward {} actions_made {}: '.format(idx, len(traj), all_rewards[0][idx - 1],
                                                                        all_actions[0][0]))
        if (args.algo == 'ppo'):
            curr_traj = traj.squeeze(1)
            for experience in curr_traj:
                print('curr_state: {} prev_action: {} prev_reward: {} is_done: {}'.format(experience[:num_states],
                                                                                          experience[
                                                                                          num_states:num_states + num_actions],
                                                                                          experience[
                                                                                              num_states + num_actions],
                                                                                          experience[-1]))
    print(all_actions)
    print(all_rewards)


if __name__ == '__main__':
    if (not args.eval):
        print("TRAINING MODEL ========================================================================")
        meta_train()
    print("TESTING MODEL ========================================================================")
    evaluate_model(args.eval_model)
