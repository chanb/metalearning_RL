import torch
import numpy as np
import gym
import multiprocessing as mp
from helper.envs.multiprocessing_env import SubprocVecEnv

EPS = 1e-8


# Reference: MAML-PyTorch, RL-Adventure, pytorch-ppo-acktr-a2c, openAI baseline, openAI spinningup


# Returns a callable function for SubprocVecEnv
def make_env(env_name):
  def _make_env():
    return gym.make(env_name)
  return _make_env


# This samples from the current environment using the provided model
class Sampler():
  def __init__(self, model, env_name, num_actions, gamma=0.99, tau=0.3, num_workers=mp.cpu_count() - 1):
    self.model = model
    self.env_name = env_name
    self.num_actions = num_actions
    self.gamma = gamma
    self.tau = tau
    self.last_hidden_state = None
    self.hidden_states = []

    # This is for multi-processing
    self.num_workers = num_workers
    self.envs = SubprocVecEnv([make_env(env_name) for _ in range(num_workers)])

    self.reset_storage()

  # Computes the advantage where lambda = tau
  def compute_gae(self, next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
      delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
      gae = delta + gamma * tau * masks[step] * gae
      returns.insert(0, gae + values[step])
    return returns


  # Set the current task
  def set_task(self, task):
    print('Env Setup:')
    print(task)
    tasks = [task for _ in range(self.num_workers)]
    reset = self.envs.reset_task(tasks)
    return all(reset)


  # Reset the storage
  def reset_storage(self):
    self.actions = []
    self.values = []
    self.states = []
    self.rewards = []
    self.log_probs = []
    self.masks = []
    self.returns = []
    self.advantages = []
    self.reset_debug()


  # Concatenate storage for more accessibility
  def concat_storage(self):
    # Store in better format
    self.returns = torch.cat(self.returns).detach()
    self.values = torch.cat(self.values).detach()
    self.log_probs = torch.cat(self.log_probs).detach()
    self.states = torch.cat(self.states)
    self.actions = torch.cat(self.actions)
    self.advantages = self.returns - self.values
    self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + EPS)


  # Concatenate hidden state
  def get_hidden_state(self):
    return torch.cat(self.hidden_states).detach()


  # Insert a sample into the storage
  def insert_storage(self, log_prob, state, action, reward, done, value, hidden_state):
      self.log_probs.append(log_prob)
      self.states.append(state)
      self.actions.append(action)
      self.rewards.append(reward)
      self.masks.append(1 - done)
      self.values.append(value)
      self.hidden_states.append(hidden_state)


  def reset_traj(self):
    states = self.envs.reset()
    return torch.from_numpy(states), torch.zeros([self.num_workers, ]), torch.from_numpy(np.full((self.num_workers, ), -1)), torch.zeros([self.num_workers, ])


  # Generate the state vector for RNN
  def generate_state_vector(self, done, reward, num_actions, action, state):
    done_entry = done.float().unsqueeze(1)
    reward_entry = reward.float().unsqueeze(1)
    action_vector = torch.zeros([self.num_workers, num_actions])
    assert all(action > -1) or all(action == -1), 'All processes should be at the same step'
    if (all(action > -1)):
      action_vector.scatter_(1, action.unsqueeze(1), 1)
    #TODO: Remove printing
    # print("tes!!!!!t")
    # print('{}\n{}\n{}\n{}'.format(done_entry, reward_entry, action_vector, state))
    
    state = torch.cat((state, action_vector, reward_entry, done_entry), 1)
    # print(state)
    state = state.unsqueeze(0)
    return state


  # Sample batchsize amount of moves
  def sample(self, batchsize, last_hidden_state=None):

    state, reward, action, done = self.reset_traj()

    #TODO: Add code to handle non recurrent case
    hidden_state = last_hidden_state
    if last_hidden_state is None:
      hidden_state = self.model.init_hidden_state(self.num_workers)

    # We sample batchsize amount of data
    for i in range(batchsize):
      # Set the vector state
      if self.model.is_recurrent:
        state = self.generate_state_vector(done, reward, self.num_actions, action, state)

      # Get information from model and take action
      with torch.no_grad():
        dist, value, next_hidden_state = self.model(state, hidden_state, to_print=False)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        next_state, reward, done, _ = self.envs.step(action.cpu().numpy())
        done = done.astype(int)
        
        reward = torch.from_numpy(reward).float()
        done = torch.from_numpy(done).float()

        print('{}\n{}\n{}'.format(next_state, reward, done))
        
      # Store the information
      self.insert_storage(log_prob, state, action, reward, done, value, hidden_state)

      ########################################################################
      # Storing this for debugging
      self.clean_actions.append(action)
      self.clean_states.append(state)
      self.clean_rewards.append(reward)
      ########################################################################

      # Update to the next value
      state = next_state
      state = torch.from_numpy(state).float()
      
      hidden_state = next_hidden_state

      # Grab hidden state for the extra information
      assert all(done) or all(not done), 'All processes be done at the same time'
      if (all(done)):
        if self.model.is_recurrent:
          state = self.generate_state_vector(done, reward, self.num_actions, action, state)

        #TODO: Remove to_print
        _, _, hidden_state = self.model(state, hidden_state, to_print=False)
        state, reward, action, done = self.reset_traj()


    ########################################################################
    # self.print_debug()
    ########################################################################
    self.last_hidden_state = hidden_state
    #TODO: Remove to_print
    # Compute the return
    if self.model.is_recurrent:
      state = self.generate_state_vector(done, reward, self.num_actions, action, state)
      with torch.no_grad():
        _, next_val, _, = self.model(state, hidden_state, to_print=False)

    self.returns = self.compute_gae(next_val, self.rewards, self.masks, self.values, self.gamma, self.tau)


  # Reset debugging information
  def reset_debug(self):
    self.clean_actions = []
    self.clean_states = []
    self.clean_rewards = []


  # Print debugging information
  def print_debug(self):
    for action, state, reward in zip(self.clean_actions, self.clean_states, self.clean_rewards):
      print('action: {} reward: {} state: {}'.format(action, reward, state))