import gym
import numpy as np
import argparse
import pickle

import torch
import torch.optim as optim

import helper.envs
from helper.algo import ppo, reinforce
import os

parser = argparse.ArgumentParser(description='Evaluate model on specified task')

parser.add_argument('--task', type=str, default='bandit', help='the task to learn [bandit, mdp] (default: bandit)')
parser.add_argument('--algo', type=str, default='ppo', help='algorithm to use [reinforce/ppo] (default: ppo)')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate for optimizer (default: 1e-2)')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor (default: 0.99)')

parser.add_argument('--num_actions', type=int, default=5, help='number of arms for MAB or number of actions for MDP (default: 5)')
parser.add_argument('--num_tasks', type=int, default=5, help='number of similar tasks to run (default: 5)')
parser.add_argument('--num_traj', type=int, default=10, help='number of trajectories to interact with (default: 10)')
parser.add_argument('--traj_len', type=int, default=1, help='fixed trajectory length (default: 1)')

parser.add_argument('--tau', type=float, default=0.95, help='GAE parameter (default: 0.95)')
parser.add_argument('--mini_batch_size', type=int, default=5, help='minibatch size for ppo update (default: 5)')
parser.add_argument('--batch_size', type=int, default=5, help='batch size (default: 5)')
parser.add_argument('--ppo_epochs', type=int, default=1, help='ppo epoch (default: 1)')
parser.add_argument('--clip_param', type=float, default=0.2, help='clipping parameter for PPO (default: 0.2)')

parser.add_argument('--eval_model', help='the model to evaluate')
parser.add_argument('--eval_tasks', help='the tasks to evaluate on')

parser.add_argument('--outfile', help='filename to save output')

args = parser.parse_args()

#result_folder = './logs_eval'
#out_result = '{}/{}_{}_{}_{}.pkl'.format(result_folder, args.algo, args.task, args.num_actions, args.max_num_traj_eval)
out_result =args.outfile

def evaluate_model(eval_model=None, eval_tasks=None):
    if (not eval_model):
        print("No model to evaluate on")
        return

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

    env = gym.make(task)
    tasks = env.unwrapped.sample_tasks(args.num_tasks)
    if (eval_tasks):
        with open(eval_tasks, 'rb') as f:
            tasks = pickle.load(f)[0]
    
    if args.algo == 'ppo':
        all_rewards, all_states, all_actions, _ = ppo(model, task, num_actions, args.num_tasks, args.num_traj, args.traj_len,
            args.ppo_epochs, args.mini_batch_size, args.batch_size, args.gamma, args.tau, args.clip_param, args.learning_rate)
    else:
        print('Invalid learning algorithm')

    with open(out_result, 'wb') as f:
        pickle.dump([all_rewards, all_actions, all_states, num_actions, num_states], f)

    # idx = 0
    # for traj in all_states[0]:
    #     idx += 1
    #     curr_traj = traj
    #     print('traj {} (length: {}) reward {} actions_made {}: '.format(idx, len(traj), all_rewards[0][idx - 1],
    #                                                                     all_actions[0][0]))
    #     if (args.algo == 'ppo'):
    #         curr_traj = traj.squeeze(1)
    #         for experience in curr_traj:
    #             print('curr_state: {} prev_action: {} prev_reward: {} is_done: {}'.format(experience[:num_states],
    #                                                                                       experience[
    #                                                                                       num_states:num_states + num_actions],
    #                                                                                       experience[
    #                                                                                           num_states + num_actions],
    #                                                                                       experience[-1]))
    # print(all_actions)
    # print(all_rewards)

if __name__ == "__main__":
  print("TESTING MODEL ========================================================================")
  evaluate_model(args.eval_model, args.eval_tasks)