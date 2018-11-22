import gym
import numpy as np
import argparse
import helper.envs
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# Computes the advantage where lambda = tau
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
        rand_ids = np.random.choice(batch_size, mini_batch_size, False)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], \
              advantages[rand_ids, :]


def ppo_update(model, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
    # Use Clipping Surrogate Objective to update
    for i in range(ppo_epochs):
        for state, action, old_log_probs, ret, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns,
                                                                advantages):
            new_log_probs = []
            for sample in range(mini_batch_size):
                dist, value = model(state[sample].unsqueeze(0), keep=False)
                m = Categorical(logits=dist)
                entropy = m.entropy().mean()
                new_log_probs.append([m.log_prob(action[sample])])

            new_log_probs = torch.tensor(new_log_probs)

            ratio = (new_log_probs - old_log_probs).exp()
            
            surr_1 = ratio * advantage
            surr_2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            # Clipped Surrogate Objective Loss
            actor_loss = torch.min(surr_1, surr_2).mean()

            # Mean Squared Error Loss Function
            critic_loss = (ret - value).pow(2).mean()#F.mse_loss(ret, value)

            # This is L(Clip) - c_1L(VF) + c_2L(S)
            # Take negative because we're doing gradient descent
            # loss = (critic_loss - actor_loss - 0.01 * entropy)
            loss = -actor_loss

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()


            print("new_log_prob: {} \nold_log_prob: {} \nactions: {} \nreturn: {}".format(new_log_probs.squeeze(1), old_log_probs.squeeze(1), action.squeeze(), ret.squeeze()))
            # print("ret: {} val: {}".format(ret.squeeze(1).squeeze(1), value.squeeze(1).squeeze(1)))
            print("action: {} return: {} \nratio: {} critic_loss: {} actor_loss: {} entropy: {} loss: {}\n".format(action.squeeze(), ret.squeeze(), ratio.squeeze(), critic_loss.squeeze(), actor_loss.squeeze(), entropy, loss.squeeze()))


def ppo_sample(env, model, num_actions, num_traj, traj_len, ppo_epochs, mini_batch_size, batch_size, gamma, tau, clip_param, learning_rate):
    '''
    env - the gym environment for the current task
    model - the model to update
    num_actions - (k) the number of possible actions
    num_traj - (n) the number of trajectories/episodes to interact with
    traj_len - (l) the fixed length of each trajectory/episode
    ppo_epochs - (K) the number of PPO updates to perform
    batch_size - (T) the number of steps to take before PPO update
    mini_batch_size - (M) the number of horizon to take for the PPO update
    gamma - the discount factor
    tau - the GAE parameter
    clip_param - The clipping parameter for L_clip
    learning_rate - The learning rate of the Adam optimizer

    The total number of horizon for current task is n x l
    The batch_size should divide n x l, in order to get number of batches
    '''

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    task_total_rewards = []
    task_total_states = []
    task_total_actions = []

    num_batches = num_traj * traj_len // batch_size

    # PPO (Using actor critic style)
    for batch in range(num_batches):
        print("Batch {} of {}".format(batch + 1, num_batches))

        log_probs = []
        values = []
        states = []
        actions = []
        rewards = []
        masks = []

        # These are for logging
        clean_actions = []
        clean_rewards = []
        clean_states = []

        # The initial values for input
        state = env.reset()
        reward = 0.
        action = -1
        done = 0

        for horizon in range(batch_size):
            if (horizon + 1 % 10 == 0):
                print('Horizon {} of {}'.format(horizon + 1, batch_size))
            state = torch.from_numpy(state).float().unsqueeze(0)

            # Construct the (s,a,r,d) as input
            if model.is_recurrent:
                done_entry = torch.tensor([[done]]).float()
                reward_entry = torch.tensor([[reward]]).float()
                action_vector = torch.FloatTensor(num_actions)
                action_vector.zero_()
                if (action > -1):
                    action_vector[action] = 1
                
                action_vector = action_vector.unsqueeze(0)
                
                state = torch.cat((state, action_vector, reward_entry, done_entry), 1)
                state = state.unsqueeze(0)

            # Sample the next action from the model
            dist, value = model(state)
            m = Categorical(logits=dist)
            action = m.sample()
            
            # Take the action
            next_state, reward, done, _ = env.step(action.item())
            print('dist: {} action: {} reward: {}'.format(F.softmax(dist, dim=0), action, reward))

            # Accumulate all the information
            done = int(done)
            log_prob = m.log_prob(action)
            log_probs.append(log_prob.unsqueeze(0).unsqueeze(0))
            clean_actions.append(action.data.item())
            clean_states.append(state)
            clean_rewards.append(reward)
            states.append(state)
            actions.append(action.unsqueeze(0).unsqueeze(0))
            rewards.append(reward)
            masks.append(1 - done)
            values.append(value)

            state = next_state

            # Reset input values if we're done the trajectory/episode
            if (done):
                state = env.reset()
                reward = 0.
                action = -1
                done = 0
                task_total_actions.append(clean_actions)
                task_total_rewards.append(sum(clean_rewards))
                task_total_states.append(clean_states)
                clean_actions = []
                clean_rewards = []
                clean_states = []

        state = torch.from_numpy(state).float().unsqueeze(0)
        if model.is_recurrent:
            done_entry = torch.tensor([[done]]).float()
            reward_entry = torch.tensor([[reward]]).float()
            action_vector = torch.FloatTensor(num_actions)
            action_vector.zero_()
            action_vector[action] = 1
            action_vector = action_vector.unsqueeze(0)
            state = torch.cat((state, action_vector, reward_entry, done_entry), 1)
            state = state.unsqueeze(0)

        next_dist, next_val = model(state, keep=False)

        returns = compute_gae(next_val, rewards, masks, values, gamma, tau)
        returns = torch.cat(returns)
        values = torch.cat(values)
        log_probs = torch.cat(log_probs)
        states = torch.cat(states)
        actions = torch.cat(actions)
        advantage = returns - values
        # advantage = (advantage - advantage.mean())/(advantage.std() + 1e-5)

        # This is where we compute loss and update the model
        ppo_update(model, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage
                    , clip_param=clip_param)

    return task_total_rewards, task_total_states, task_total_actions
    


# Attempt to modify policy so it doesn't go too far
def ppo(model, rl_category, num_actions, num_tasks, num_traj, traj_len, ppo_epochs, mini_batch_size, batch_size,
        gamma, tau, clip_param, learning_rate, evaluate_tasks=None, evaluate_model=None):
    all_rewards = []
    all_states = []
    all_actions = []

    # Meta-Learning on a class of MDP problem
    env = gym.make(rl_category)
    tasks = evaluate_tasks

    # Sample a specified amount of tasks from the class of MDP if we aren't provided any tasks
    if (not evaluate_tasks):
        tasks = env.unwrapped.sample_tasks(num_tasks)

    # Learn on every sampled task
    for task in range(len(tasks)):
        if((task + 1) % 10 == 0):
            print(
              "Task {} ==========================================================================================================".format(
                task + 1))

        # Reload the model if we're evaluating model
        if (evaluate_model):
            policy = torch.load(evaluate_model)

        # Need to reset hidden state for every new task
        if model.is_recurrent:
            model.reset_hidden_state()

        # Update the environment to use the new task
        env.unwrapped.reset_task(tasks[task])

        # Perform sampling and update model
        task_total_rewards, task_total_states, task_total_actions = ppo_sample(env, model, num_actions, num_traj, traj_len, ppo_epochs, mini_batch_size, batch_size, gamma, tau, clip_param, learning_rate)
        
        all_rewards.append(task_total_rewards)
        all_states.append(task_total_states)
        all_actions.append(task_total_actions)

    return all_rewards, all_states, all_actions, model
