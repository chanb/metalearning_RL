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

parser.add_argument('--algo', type=str, default='reinforce',
                    help='algorithm to use [reinforce] (default: reinforce)')
parser.add_argument('--learning_rate', type=float, default=1e-2,
                    help='learning rate for gradient descent (default: 1e-2)')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor (default: 0.99)')

parser.add_argument('--task', type=str, default='bandit', help='the task to learn [bandit, mdp] (default: bandit)')
parser.add_argument('--num_actions', type=int, default=5,
                    help='number of arms for MAB or number of actions for MDP (default: 5)')

parser.add_argument('--num_tasks', type=int, default=5, help='number of similar tasks to run (default: 5)')
parser.add_argument('--max_num_traj_eval', type=int, default=1000,
                    help='maximum number of trajectories during evaluation (default: 1000)')
parser.add_argument('--max_traj_len', type=int, default=1, help='maximum trajectory length (default: 1)')


parser.add_argument('--eval_model', help='the model to evaluate')
parser.add_argument('--eval_tasks', help='the tasks to evaluate on')

args = parser.parse_args()

result_folder = './logs'
out_result = '{}/{}_{}_{}_{}.pkl'.format(result_folder, args.algo, args.task, args.num_actions, args.max_num_traj_eval)

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
    print(len(tasks))
    
    if args.algo == 'reinforce':
        all_rewards, all_states, all_actions, _ = reinforce(model, optimizer, task, num_actions, args.num_tasks,
                                                            args.max_num_traj_eval, args.max_traj_len,
                                                            args.gamma, evaluate_tasks=tasks)
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

if __name__ == "__main__":
  print("TESTING MODEL ========================================================================")
  evaluate_model(args.eval_model, args.eval_tasks)