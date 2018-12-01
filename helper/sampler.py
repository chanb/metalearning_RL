import torch

eps = 1e-5

# This samples from the current environment using the provided model
class Sampler():
  def __init__(self, model, env, num_actions, gamma=0.99, tau=0.3):
    self.model = model
    self.env = env
    self.num_actions = num_actions
    self.gamma = gamma
    self.tau = tau

    self.reset_storage()
    self.reset_debug()
    

  # Reset the current environment
  def set_env(self, task):
    self.env.unwrapped.reset_task(task)

  # Reset the storage
  def reset_storage(self):
    self.actions = []
    self.values = []
    self.states = []
    self.rewards = []
    self.log_probs = []
    self.masks = []
    self.hidden_states = []
    self.returns = []
    self.advantages = []

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

  # Generate the state vector for RNN
  def generate_state_vector(self, done, reward, num_actions, action, state):
    done_entry = torch.tensor([[done]]).float()
    reward_entry = torch.tensor([[reward]]).float()
    action_vector = torch.FloatTensor(num_actions)
    action_vector.zero_()
    if (action > -1):
      action_vector[action] = 1
    
    action_vector = action_vector.unsqueeze(0)
    state = torch.cat((state, action_vector, reward_entry, done_entry), 1)
    state = state.unsqueeze(0)
    return state

  # Sample batchsize amount of moves
  def sample(self, batchsize):
    state = self.env.reset()
    state = torch.from_numpy(state).float().unsqueeze(0)
    reward = 0
    action = -1
    done = 0

    hidden_state = torch.zeros([1, 1, self.model.hidden_size])

    for i in range(batchsize):
      # Set the vector state
      if self.model.is_recurrent:
        state = self.generate_state_vector(done, reward, self.num_actions, action, state)

      # Get information from model and take action
      dist, value, next_hidden_state = self.model(state, hidden_state)
      action = dist.sample()
      log_prob = dist.log_prob(action)
      next_state, reward, done, _ = self.env.step(action.item())

      ########################################################################
      # Storing this for debugging
      self.clean_actions.append(action.data.item())
      self.clean_states.append(state)
      self.clean_rewards.append(reward)
      ########################################################################

      # Store the information
      self.log_probs.append(log_prob.unsqueeze(0).unsqueeze(0))
      self.states.append(state)
      self.actions.append(action.unsqueeze(0).unsqueeze(0))
      self.rewards.append(reward)
      self.masks.append(1 - done)
      self.values.append(value)
      self.hidden_states.append(hidden_state)

      state = next_state
      state = torch.from_numpy(state).float().unsqueeze(0)
      hidden_state = next_hidden_state

      # Grab hidden state for the extra information
      if (done):
        if self.model.is_recurrent:
          state = self.generate_state_vector(done, reward, self.num_actions, action, state)

        #TODO: Remove to_print
        _, _, hidden_state = self.model(state, hidden_state, to_print=False)
        state = self.env.reset()
        state = torch.from_numpy(state).float().unsqueeze(0)
        reward = 0
        action = -1
        done = 0

    ########################################################################
    # self.print_debug()
    ########################################################################
    
    #TODO: Remove to_print
    # Compute the return
    if self.model.is_recurrent:
      state = self.generate_state_vector(done, reward, self.num_actions, action, state)
      _, next_val, _, = self.model(state, hidden_state, to_print=False)

    self.returns = self.compute_gae(next_val, self.rewards, self.masks, self.values, self.gamma, self.tau)
  
    # Store in better format
    self.returns = torch.cat(self.returns)
    self.values = torch.cat(self.values)
    self.log_probs = torch.cat(self.log_probs)
    self.states = torch.cat(self.states)
    self.actions = torch.cat(self.actions)
    self.hidden_states = torch.cat(self.hidden_states)
    self.advantages = self.returns - self.values
    self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + eps)

  # Reset debugging information
  def reset_debug(self):
    self.clean_actions = []
    self.clean_states = []
    self.clean_rewards = []

  # Print debugging information
  def print_debug(self):
    for action, state, reward in zip(self.clean_actions, self.clean_states, self.clean_rewards):
      print('action: {} reward: {} state: {}'.format(action, reward, state))