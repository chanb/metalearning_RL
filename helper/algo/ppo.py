import gym
import numpy as np
import argparse
import helper.envs
import torch
import torch.optim as optim
from torch.distributions import Categorical

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
def ppo(model, optimizer, rl_category, num_actions, num_tasks, max_num_traj, max_traj_len, ppo_epochs, mini_batch_size):
    all_rewards = []

    # Meta-Learning
    for task in range(num_tasks):
        task_total_rewards = []
        if (task % 50 == 0):
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

            # print("DATA =====================")
            # print(returns)
            # print(values)
            # print(advantage)
            # print(actions)
            # print(states)
            # print(log_probs)
            # print(rewards)
            task_total_rewards.append(sum(rewards))

            # This is where we compute loss and update the model
            ppo_update(model, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage)

        all_rewards.append(sum(task_total_rewards))
        if model.is_recurrent:
            model.reset_hidden_state()
    return all_rewards, model
