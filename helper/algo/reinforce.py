import gym
import numpy as np
import argparse
import helper.envs
import torch
import torch.optim as optim
from torch.distributions import Categorical

def select_action(policy, state, traj):
    state = torch.from_numpy(state).float().unsqueeze(0)

    if policy.is_recurrent:
        state = state.unsqueeze(0)

    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def reinforce(policy, optimizer, rl_category, num_actions, num_tasks, max_num_traj, max_traj_len, discount_factor):
    # TODO: Add randomize number of trajectories to run
    all_rewards = []
    all_states = []
    all_actions = []

    # Meta-Learning
    for task in range(num_tasks):
        task_total_rewards = []
        task_total_actions = []
        task_total_states = []
        print(
            "Task {} ==========================================================================================================".format(
                task))
        env = gym.make(rl_category)

        # REINFORCE
        for traj in range(max_num_traj):
            if (traj % 50 == 0):
                print("Trajectory {}".format(traj))
            state = env.reset()

            rewards = []
            actions = []
            states = []
            for horizon in range(max_traj_len):
                action = select_action(policy, state, traj)
                state, reward, done, info = env.step(action)

                actions.append(action)
                rewards.append(reward)
                states.append(state)
                if (done):
                    break

            # Batch gradient descent
            R = 0
            policy_loss = []
            discounted_rewards = []
            traj_len = len(discounted_rewards)

            for r in rewards[::-1]:
                R = r + discount_factor * R
                discounted_rewards.insert(0, R)

            discounted_rewards = torch.tensor(discounted_rewards)

            if (traj_len > 1):
                discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + eps)

            # Compute loss and take gradient step
            for log_prob, reward in zip(policy.saved_log_probs, discounted_rewards):
                policy_loss.append(-log_prob * reward)

            optimizer.zero_grad()
            policy_loss = torch.cat(policy_loss).sum()
            
            policy_loss.backward(retain_graph=policy.is_recurrent)
            optimizer.step()
            del policy.saved_log_probs[:]

            task_total_rewards.append(sum(rewards))
            task_total_states.append(states)
            task_total_actions.append(actions)

        all_rewards.append(task_total_rewards)
        all_actions.append(task_total_actions)
        all_states.append(task_total_states)
        if policy.is_recurrent:
            policy.reset_hidden_state()
    return all_rewards, all_states, all_actions, policy
