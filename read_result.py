import pickle
import argparse
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Reads output file from running MDP/MAB using RL2')

parser.add_argument('--file', type=str, help='output file to read from')
parser.add_argument('--algo', type=str, help='the algorithm used to generate the output')

args = parser.parse_args()

with open(args.file, 'rb') as f:  # Python 3: open(..., 'rb')
    all_rewards, all_actions, all_states, num_actions, num_states = pickle.load(f)

    # convert to a numpy matrix
    all_rewards_matrix = np.array([np.array(xi) for xi in all_rewards])
    # each row now contains values for each iteration
    all_rewards_matrix = all_rewards_matrix.T
    # computes std dev of each row
    #reward_err = np.std(all_rewards_matrix, axis=1)
    avg_reward = np.average(all_rewards_matrix, axis=1)
    # # normalizing
    # avg_reward = (avg_reward - (avg_reward.min()))/(avg_reward.max() - avg_reward.min())
    # reward_err = (reward_err - (reward_err.min()))/(reward_err.max() - reward_err.min())

    #print(avg_reward / 100)

    # plotting
    plt.plot(range(len(avg_reward)), avg_reward)
   #plt.plot(x=len(avg_reward), y=reward_err)
    plt.xlabel('Number of Iterations')
    plt.ylabel('Total Reward')
    plt.title('Model Performance')
   #plt.fill_between(x=range(len(avg_reward)), y1=reward_err, color = 'gray')
    plt.show()

    # print(all_rewards)
    # print(all_actions)


