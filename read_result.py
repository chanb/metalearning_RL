import pickle
import argparse

parser = argparse.ArgumentParser(description='Reads output file from running MDP/MAB using RL2')

parser.add_argument('--file', type=str, help='output file to read from')
parser.add_argument('--algo', type=str, help='the algorithm used to generate the output')

args = parser.parse_args()

with open(args.file, 'rb') as f:  # Python 3: open(..., 'rb')
    all_rewards, all_actions, all_states, num_actions, num_states = pickle.load(f)

    idx = 0 
    for traj in all_states[0]:
        idx += 1
        curr_traj = traj
        print('traj {} (length: {}) reward {} actions_made {}: '.format(idx, len(traj), all_rewards[0][idx - 1], all_actions[0][0]))
        if (args.algo == 'ppo'):
            curr_traj = traj.squeeze(1)
            for experience in curr_traj:
                print('curr_state: {} prev_action: {} prev_reward: {} is_done: {}'.format(experience[:num_states], experience[num_states:num_states + num_actions], experience[num_states + num_actions], experience[-1]))


    print(all_rewards)
    print(all_actions)