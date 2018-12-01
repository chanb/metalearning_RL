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
      print('{} {} {}'.format(curr_traj, self.num_traj, curr_batchsize))
      if curr_traj == 0:
        self.set_env(curr_task)
        curr_task += 1
      
      # If the whole task can fit, sample the whole task
      if curr_batchsize + (self.num_traj - curr_traj) <= agent.batchsize:
        print('yes')
        sample_amount = self.num_traj - curr_traj
        sampler.sample(sample_amount)
        sampler.last_hidden_state = None
      else:
        print('no')
        sample_amount = agent.batchsize - curr_batchsize
        sampler.sample(sample_amount, sampler.last_hidden_state)
      print(sample_amount)
      i += sample_amount
      curr_batchsize += sample_amount
      curr_traj += sample_amount

      if curr_batchsize == agent.batchsize:
        sampler.concat_storage()
        agent.update(sampler)
        sampler.reset_storage()
        curr_batchsize = 0

      if curr_traj == self.num_traj:
        curr_traj = 0
      
    if total_num_steps % agent.batchsize != 0:
      sampler.concat_storage()
      agent.update(sampler)
      sampler.reset_storage()
        

      