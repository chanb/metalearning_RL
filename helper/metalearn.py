import gym
from helper.sampler import Sampler
from helper.algo import PPO

class MetaLearner():
  def __init__(self, task, num_actions, num_states, num_tasks, num_traj, traj_len):
    self.num_actions = num_actions
    self.num_states = num_states
    self.num_tasks = num_tasks
    self.num_traj = num_traj
    self.traj_len = traj_len

    self.env = gym.make(task)
    self.tasks = self.env.unwrapped.sample_tasks(num_tasks)

  # Resample the tasks
  def sample_tasks(self):
    self.tasks = self.env.unwrapped.sample_tasks(self.num_tasks)

  # Set the environment using the i'th task
  def set_env(self, i):
    if (i >= self.num_tasks or i < 0):
      assert (i < self.num_tasks and i >= 0), 'i = {} is out of range. There is only {} tasks'.format(i, self.num_tasks)
    self.env.unwrapped.reset_task(self.tasks[i])

  def train(self, model, optimizer, agent, gamma, tau):
    sampler = Sampler(model, self.env, self.num_actions, gamma, tau)

    total_num_steps = self.num_traj * self.traj_len * self.num_tasks

    curr_traj = 0
    curr_batchsize = 0
    curr_task = 0
    i = 0

    while i < total_num_steps:
      if curr_traj == 0:
        self.set_env(curr_task)
        curr_task += 1
      
      # Sample batch size
      if curr_traj + agent.batchsize <= self.num_traj:
        sampler.sample(agent.batchsize)
        i += agent.batchsize
        curr_traj += agent.batchsize
        curr_batchsize = agent.batchsize
      else:
        sampler.sample(self.num_traj - curr_traj, sampler.last_hidden_state)
        i += self.num_traj - curr_traj
        curr_batchsize += (self.num_traj - curr_traj)
        curr_traj = self.num_traj

      if curr_batchsize == agent.batchsize:
        sampler.concat_storage()
        agent.update(sampler)
        sampler.reset_storage()
        curr_batchsize = 0

      if curr_traj == self.num_traj:
        curr_traj = 0
        

      