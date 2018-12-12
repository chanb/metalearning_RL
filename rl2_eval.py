from multiprocessing.pool import ThreadPool
import pickle
import argparse
import glob
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')

import matplotlib.pyplot as plt
import torch

from helper.evaluate_model import evaluate_multiple_tasks, sample_multiple_random_fixed_length


parser = argparse.ArgumentParser(description='Evaluate model on specified task')

parser.add_argument('--task', type=str, default='bandit', help='the task to learn [bandit, mdp] (default: bandit)')
parser.add_argument('--algo', type=str, default='ppo', help='the algorithm to evaluate (default: ppo)')

parser.add_argument('--num_actions', type=int, default=5, help='number of arms for MAB or number of actions for MDP (default: 5)')
parser.add_argument('--num_tasks', type=int, default=100, help='number of similar tasks to run (default: 100)')
parser.add_argument('--num_traj', type=int, default=10, help='number of trajectories to interact with (default: 10)')
parser.add_argument('--traj_len', type=int, default=1, help='fixed trajectory length (default: 1)')

parser.add_argument('--num_fake_update', type=int, default=300, help='number of fake gradient updates. used by random sampling (default: 300)')
parser.add_argument('--num_workers', type=int, default=3, help='number of workers to perform evaluation in parallel (default: 3)')

parser.add_argument('--models_dir', help='the directory of the models to evaluate. models are retrieved in increasing order based on number prefix')
parser.add_argument('--eval_tasks', help='the tasks to evaluate on')
parser.add_argument('--out_file', help='the prefix of the filename to save outputs')

args = parser.parse_args()


def evaluate_result(algo, env_name, tasks, num_actions, num_traj, traj_len, models_dir, out_file_prefix, num_workers=3, num_fake_update=300):
  pool = ThreadPool(processes=num_workers)

  if algo == 'ppo':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = glob.glob('./{0}/*_{0}.pt'.format(models_dir))
    results = [pool.apply(evaluate_multiple_tasks, args=(device, env_name, model, tasks, num_actions, num_traj, traj_len)) for model in models]
  else:
    results = [pool.apply(sample_multiple_random_fixed_length, args=(env_name, tasks, num_actions, num_traj, traj_len)) for model in range(num_fake_update)]

  assert not results or len(results == 0), 'results should not be empty'

  all_rewards, all_actions, all_states = zip(*results)

  # saves all rewards, actions, and states to a new file for later plotting all on one graph
  with open('{}.pkl'.format(out_file_prefix), 'wb') as pickle_out:
    pickle.dump([all_rewards, all_actions, all_states], pickle_out)


def generate_plot(out_file_prefix):
  with open('{}.pkl'.format(out_file_prefix), 'rb') as f:
    all_rewards, _, _ = pickle.load(f)

  all_rewards_matrix = np.array([np.array(curr_model_rewards) for curr_model_rewards in all_rewards])

  # Compute the average and standard deviation of each model over specified number of tasks
  models_avg_rewards = np.average(all_rewards_matrix, axis=1)
  models_std_rewards = np.std(all_rewards_matrix, axis=1)
  
  plt.plot(range(1, len(models_avg_rewards) + 1), models_avg_rewards)
  plt.xlabel('Number of Updates')
  plt.ylabel('Average Total Reward')
  plt.title('Model Performance')

  plt.fill_between(range(1, len(models_avg_rewards) + 1), models_avg_rewards-models_std_rewards, models_avg_rewards+models_std_rewards, color = 'blue', alpha=0.3, lw=0.001)
  plt.savefig('{}.png'.format(out_file_prefix))

def main():
  print("TESTING MODEL ========================================================================")
  assert args.out_file, 'Missing output file'
  assert args.eval_tasks, 'Missing tasks'
  assert args.num_fake_update > 0, 'Needs to have at least 1 update'
  assert args.num_workers > 0, 'Needs to have at least 1 worker'
  assert (args.algo != 'ppo' or args.models_dir), 'Missing models'
  assert (args.algo == 'ppo' or args.algo == 'random'), 'Invalid algorithm'
  assert (args.task == 'bandit' or args.task == 'mdp'), 'Invalid task'
  env_name = ''
  if args.task == 'bandit':
    env_name = "Bandit-K{}-v0".format(args.num_actions)
    num_actions = args.num_actions
    num_states = 1
  elif args.task == 'mdp':
    env_name = "TabularMDP-v0"
    num_actions = 5
    num_states = 10

  with open(args.eval_tasks, 'rb') as f:
    tasks = pickle.load(f)[0]

  evaluate_result(args.algo, env_name, tasks, num_actions, args.num_traj, args.traj_len, args.models_dir, args.out_file, args.num_workers, args.num_fake_update)

  generate_plot(args.out_file)

if __name__ == '__main__':
  main()