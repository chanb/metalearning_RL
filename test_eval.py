# from helper.evaluate_model import sample_random_fixed_length, evaluate_multiple_tasks
# import pickle
# import torch

# num_actions = 5
# num_traj = 10
# traj_len = 10

# bandit_experiments = './experiments/bandit_{}_100.pkl'.format(num_actions)
# tabular_experiments = './experiments/mdp_5_100.pkl'


# with open(tabular_experiments, 'rb') as f:
#   tasks = pickle.load(f)[0]

# bandit_env_name = 'Bandit-K{}-v0'.format(num_actions)
# tabular_env_name = 'TabularMDP-v0'

# bandit_env_model = 'train_gru_5bandit_10traj_10tasks.pt'
# tabular_eval_model = '261_train_gru_mdp_10traj_100tasks.pt'

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# rewards, actions, states = sample_random_fixed_length(tabular_env_name, tasks, num_actions, num_traj, traj_len)
# print(len(rewards))
# # print(evaluate_multiple_tasks(device, tabular_env_name, tabular_eval_model, tasks, num_actions, num_traj, traj_len))

from rl2_eval import generate_plot
generate_plot('gru_bandit_eval')