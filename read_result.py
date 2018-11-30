import pickle
import argparse
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Reads output file from running MDP/MAB using RL2')

parser.add_argument('--file', type=str, help='output file to read from')
parser.add_argument('--algo', type=str, help='the algorithm used to generate the output')
parser.add_argument('--task', type=str, help='either bandit or mdp')
parser.add_argument('--outfile', type=str, help='writes rews and err to a new file')
args = parser.parse_args()

with open(args.file, 'rb') as f:  # Python 3: open(..., 'rb')
    all_rewards, all_actions, all_states, num_actions, num_states = pickle.load(f)
    # convert to a numpy matrix
    all_rewards_matrix = np.array([np.array(xi) for xi in all_rewards])
    print(all_rewards_matrix)
    print(np.average(np.average(all_rewards_matrix, axis=1)))
    # each row now contains values for each iteration
    all_rewards_matrix = all_rewards_matrix.T
    one_task = all_rewards_matrix[1][:]

    if(args.task == 'bandit'):
        # all_rewards_matrix = np.cumsum(all_rewards_matrix, axis=0)
        # computes std dev of each row
        reward_err = np.std(all_rewards_matrix, axis=1)
        avg_reward = np.average(all_rewards_matrix, axis=1)
        # normalizing
        # for i in range(len(avg_reward)):
        #     avg_reward[i] = avg_reward[i]/(i+1)
        #     reward_err[i] = reward_err[i]/(i+1) #+ avg_reward[i]
    elif(args.task == 'mdp'):
        reward_err = np.std(all_rewards_matrix, axis=1)
        avg_reward = np.average(all_rewards_matrix, axis=1)

    # plotting
    plt.plot(range(len(avg_reward)), avg_reward)
    #plt.plot(range(len(one_task)), one_task)
    plt.xlabel('Number of Updates')
    plt.ylabel('Total Reward')
    plt.title('Model Performance')
    # plt.fill_between(range(len(avg_reward)), avg_reward-reward_err, avg_reward+reward_err, color = 'blue', alpha=0.3, lw=0.001)
    plt.savefig('{}.png'.format(args.outfile))

# saves rews and err to a new file for later plotting all on one graph
with open('{}.pkl'.format(args.outfile), 'wb') as pickle_out:
    pickle.dump([avg_reward, reward_err], pickle_out)


