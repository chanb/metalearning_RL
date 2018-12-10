import gym
import numpy as np
import argparse
import pickle

import torch

import helper.envs
import os

from helper.sampler import Sampler

parser = argparse.ArgumentParser(description='Evaluate model on specified task')

parser.add_argument('--task', type=str, default='bandit', help='the task to learn [bandit, mdp] (default: bandit)')
parser.add_argument('--algo', type=str, default='ppo', help='the algorithm to evaluate (default: ppo)')

parser.add_argument('--num_actions', type=int, default=5, help='number of arms for MAB or number of actions for MDP (default: 5)')
parser.add_argument('--num_tasks', type=int, default=100, help='number of similar tasks to run (default: 100)')
parser.add_argument('--num_traj', type=int, default=10, help='number of trajectories to interact with (default: 10)')
parser.add_argument('--traj_len', type=int, default=1, help='fixed trajectory length (default: 1)')

parser.add_argument('--eval_model', help='the model to evaluate')
parser.add_argument('--eval_tasks', help='the tasks to evaluate on')

parser.add_argument('--out_file', help='filename to save output')

args = parser.parse_args()

out_result = args.out_file

#TODO: Make it work with new sampler
def evaluate_model(env_name, eval_model, tasks, num_actions, num_states, num_traj, traj_len):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  all_rewards = []
  all_actions = []
  all_states = []
  curr_task = 1
  for task in tasks:
    if curr_task % 10 == 0:
      print("task {} ==========================================================".format(curr_task))
    model = torch.load(eval_model).to(device)
    sampler = Sampler(device, model, env_name, num_actions, deterministic=False, num_workers=1)

    sampler.set_task(task)
    sampler.sample(num_traj * traj_len)

    task_total_rewards = []
    task_total_states = []
    task_total_actions = []

    for i in range(num_traj):
      clean_actions = []
      clean_states = []
      clean_rewards = []

      for j in range(traj_len):
        curr_idx = i * j + i
        clean_actions.append(sampler.clean_actions[curr_idx])
        clean_states.append(sampler.clean_states[curr_idx].squeeze(0).squeeze(0))
        clean_rewards.append(sampler.clean_rewards[curr_idx])
        
      task_total_rewards.append(sum(clean_rewards))
      task_total_states.append(clean_states)
      task_total_actions.append(clean_actions)

    all_rewards.append(task_total_rewards)
    all_actions.append(task_total_actions)
    all_states.append(task_total_states)
    sampler.envs.close()
    curr_task += 1

  with open(out_result, 'wb') as f:
      pickle.dump([all_rewards, all_actions, all_states, num_actions, num_states], f)

def random_arm_pull(env, num_actions, num_tasks, num_traj, tasks, num_update=20):
  all_rewards = []

  # Learn on every sampled task
  for task in range(len(tasks)):
    if((task + 1) % 10 == 0):
      print(
        "Task {} ==========================================================================================================".format(
        task + 1))

    # Update the environment to use the new task
    env.unwrapped.reset_task(tasks[task])

    update_rewards = []
    # Imitate the number of grad update
    for _ in range(num_update):
      rewards = 0
      # Pull num_traj amount of times
      for _ in range(num_traj):
        env.reset()
        action = np.random.randint(0, num_actions)
        _, reward, _, _ = env.step(action)
        rewards += reward
      update_rewards.append(rewards)

    all_rewards.append(update_rewards)

  with open(out_result, 'wb') as f:
    pickle.dump([all_rewards, 0, 0, 0, 0], f)

def main():
  print("TESTING MODEL ========================================================================")
  assert args.out_file, 'Missing output file'
  assert args.eval_tasks, 'Missing tasks'
  assert (args.algo != 'ppo' or args.eval_model), 'Missing models'
  assert (args.task == 'bandit' or args.task == 'mdp'), 'Invalid Task'
  task = ''
  if args.task == 'bandit':
    task = "Bandit-K{}-v0".format(args.num_actions)
    num_actions = args.num_actions
    num_states = 1
  elif args.task == 'mdp':
    task = "TabularMDP-v0"
    num_actions = 5
    num_states = 10
  
  env = gym.make(task)

  with open(args.eval_tasks, 'rb') as f:
    tasks = pickle.load(f)[0]

  if args.algo == 'ppo':
    evaluate_model(task, args.eval_model, tasks, num_actions, num_states, args.num_traj, args.traj_len)
  else:
    random_arm_pull(env, args.num_actions, args.num_tasks, args.num_traj, tasks)

if __name__ == "__main__":
  main()
