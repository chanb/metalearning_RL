import gym
import numpy as np
import argparse

import torch
import torch.optim as optim
from torch.distributions import Categorical

import helper.envs
from helper.policies import SNAILPolicy, FCNPolicy
from helper.models import GRUActorCritic

parser = argparse.ArgumentParser(description='SNAIL for MAB and MDP')

parser.add_argument('--num_actions', type=int, default=5,
                    help='number of arms for MAB or number of actions for MDP (default: 5)')
parser.add_argument('--max_num_traj', type=int, default=10, help='maximum number of trajectories to run (default: 10)')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
parser.add_argument('--max_traj_len', type=int, default=1, help='maximum trajectory length (default: 1)')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor (default: 0.99)')
parser.add_argument('--learning_rate', type=float, default=1e-2,
                    help='learning rate for gradient descent (default: 1e-2)')
parser.add_argument('--num_tasks', type=int, default=5, help='number of similar tasks to run (default: 5)')
parser.add_argument('--algo', type=str, default='reinforce',
                    help='algorithm to use [reinforce/ppo] (default: reinforce)')
parser.add_argument('--mini_batch_size', type=int, default=1,
                    help='minimum batch size (default: 5) - needs to be <= max_traj_len')
parser.add_argument('--ppo_epochs', type=int, default=1, help='ppo epoch (default: 1)')
parser.add_argument('--task', type=str, default='bandit', help='the task to learn [bandit, mdp] (default: bandit)')

args = parser.parse_args()

# Create environment and initialize seed
# env.seed(args.seed)
# torch.manual_seed(args.seed)
eps = np.finfo(np.float32).eps.item()


def select_action(policy, state):
    state = torch.from_numpy(state).float().unsqueeze(0)

    if policy.is_recurrent:
        state = state.unsqueeze(0)

    probs = policy(state)
    m = Categorical(probs)
    # print(m.probs)
    action = m.sample()
    # print(action)
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def reinforce(rl_category, num_actions, opt_learning_rate, num_tasks, max_num_traj, max_traj_len, discount_factor):
    # TODO: Add randomize number of trajectories to run

    snail_policy = SNAILPolicy(num_actions, max_num_traj*max_traj_len)
    snail_optimizer = optim.Adam(snail_policy.parameters(), lr=opt_learning_rate)

    # Meta-Learning
    for task in range(num_tasks):
        print(
            "Task {} ==========================================================================================================".format(
                task))
        env = gym.make(rl_category)
        fcn_policy = FCNPolicy(num_actions, hidden_size=32) # Reset the FCN for each task
        fcn_optimizer = optim.Adam(fcn_policy.parameters(), lr=opt_learning_rate)
        policy_losses = []
        # REINFORCE
        states_set = []
        actions_set = []
        rewards_set = []
        for traj in range(max_num_traj):
            state = env.reset()

            rewards = []
            actions = []
            for horizon in range(max_traj_len):
                action = select_action(fcn_policy, state)
                state, reward, done, info = env.step(action)

                actions.append(action)
                rewards.append(reward)

                states_set.append(state[0])
                actions_set.append(action)
                rewards_set.append(reward)
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
            for log_prob, reward in zip(fcn_policy.saved_log_probs, discounted_rewards):
                policy_loss.append(-log_prob * reward)
            fcn_optimizer.zero_grad()
            policy_loss = torch.cat(policy_loss).sum()
            policy_losses.append(policy_loss)
            policy_loss.backward()
            fcn_optimizer.step()
            del fcn_policy.saved_log_probs[:]

            print(actions)
            print(rewards)

            print('Episode {}\tLast length: {:5d}\tTask: {}'.format(traj, horizon, task))

            snail_policy(states_set, actions_set, rewards_set)
            snail_optimizer.zero_grad()
            # Not sure what to do here.
            snail_optimizer.step()




# Computes the advantage where lambda = 1
def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        # print(rand_ids)
        # print(states)
        # print(actions)
        # print(log_probs)
        # print(returns)
        # print(advantages)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantages[
                                                                                                       rand_ids, :]


def ppo_update(model, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages,
               clip_param=0.2):
    # Use Clipping Surrogate Objective to update
    for i in range(ppo_epochs):
        for state, action, log_prob, ret, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns,
                                                                advantages):
            dist, value = model(state)

            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - log_probs).exp()
            surr_1 = ratio * advantage
            surr_2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            # Clipped Surrogate Objective Loss
            actor_loss = torch.min(surr_1, surr_2).mean()
            # Squared Loss Function
            critic_loss = (ret - value).pow(2).mean()

            # print('ret and value')
            # print(ret)
            # print(value)

            # This is L(Clip) + L(VF) + L(S)
            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

            # print("loss")
            # print(loss)
            # print(critic_loss)
            # print(actor_loss)
            # print(entropy)

            optimizer.zero_grad()
            loss.backward(retain_graph=model.is_recurrent)
            optimizer.step()


# Attempt to modify policy so it doesn't go too far
def ppo(rl_category, num_actions, opt_learning_rate, num_tasks, max_num_traj, max_traj_len, ppo_epochs, mini_batch_size):
    model = GRUActorCritic(num_actions, torch.randn(1, 1, 256))
    optimizer = optim.Adam(model.parameters(), lr=opt_learning_rate)

    # Meta-Learning
    for task in range(num_tasks):
        print(
            "Task {} ==========================================================================================================".format(
                task))
        env = gym.make(rl_category)

        # PPO (Using actor critic style)
        for _ in range(max_num_traj):
            state = env.reset()

            log_probs = []
            values = []
            states = []
            actions = []
            rewards = []
            masks = []
            entropy = 0

            for _ in range(max_traj_len):
                state = torch.from_numpy(state).float().unsqueeze(0)

                if model.is_recurrent:
                    state = state.unsqueeze(0)

                states.append(state)

                dist, value = model(state)
                # print(dist.probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                state, reward, done, _ = env.step(action.item())

                entropy += dist.entropy().mean()

                log_probs.append(log_prob.unsqueeze(0).unsqueeze(0))
                actions.append(action.unsqueeze(0).unsqueeze(0))
                values.append(value)
                rewards.append(reward)
                masks.append(1 - done)

                if (done):
                    break

            state = torch.from_numpy(state).float().unsqueeze(0)
            if model.is_recurrent:
                state = state.unsqueeze(0)

            _, next_val = model(state)
            returns = compute_gae(next_val, rewards, masks, values)

            # print(values)
            # print(returns)
            # print(log_probs)
            # print(states)

            returns = torch.cat(returns)
            values = torch.cat(values)
            log_probs = torch.cat(log_probs)
            states = torch.cat(states)
            actions = torch.cat(actions)

            advantage = returns - values

            print("DATA =====================")
            # print(returns)
            # print(values)
            # print(advantage)
            print(actions)
            # print(states)
            # print(log_probs)
            print(rewards)

            # This is where we compute loss and update the model
            ppo_update(model, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage)

        if model.is_recurrent:
            model.reset_hidden_state()


def main():
    task = ''
    if args.task == 'bandit':
        task = "Bandit-K{}-v0".format(args.num_actions)
        num_actions = args.num_actions
    elif args.task == 'mdp':
        task = "TabularMDP-v0"
        num_actions = 5
    else:
        print('Invalid Task')
        return
    if args.algo == 'reinforce':
        reinforce(task, num_actions, args.learning_rate, args.num_tasks, args.max_num_traj, args.max_traj_len,
                  args.gamma)
    elif args.algo == 'ppo':
        ppo(task, num_actions, args.learning_rate, args.num_tasks, args.max_num_traj, args.max_traj_len,
            args.ppo_epochs, args.mini_batch_size)
    else:
        print('Invalid learning algorithm')


if __name__ == '__main__':
    main()
