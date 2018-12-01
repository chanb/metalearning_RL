import gym

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
    self.tasks = self.env.unwrapped.sample_tasks(num_tasks)

  # Set the environment using the i'th task
  def set_env(self, i):
    if (i >= self.num_tasks or i < 0):
      assert (i < self.num_tasks and i >= 0), 'i = {} is out of range. There is only {} tasks'.format(i, self.num_tasks)
    self.env.unwrapped.reset_task(self.tasks[i])

  def sample(self):
    pass