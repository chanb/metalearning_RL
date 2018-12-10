import numpy as np

import gym
from gym import spaces
from gym.utils import seeding


class TabularMDPEnv(gym.Env):
    """Tabular MDP problems, as described in [1].

    At each time step, the agent chooses one of `num_actions` actions, say `i`, 
    receives a reward sampled from a Normal distribution with mean `m_i` and 
    variance 1 (fixed across all tasks), and reaches a new state following the 
    dynamics of the Markov Decision Process (MDP). The tabular MDP tasks are 
    generated by sampling the mean rewards from a Normal distribution with mean 
    1 and variance 1, and sampling the transition probabilities from a uniform 
    Dirichlet distribution (ie. with parameter 1).

    [1] Yan Duan, John Schulman, Xi Chen, Peter L. Bartlett, Ilya Sutskever,
        Pieter Abbeel, "RL2: Fast Reinforcement Learning via Slow Reinforcement
        Learning", 2016 (https://arxiv.org/abs/1611.02779)
    """

    def __init__(self, num_states, num_actions, task={}):
        super(TabularMDPEnv, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions

        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Box(low=0.0,
                                            high=1.0, shape=(num_states,), dtype=np.float32)

        self._task = task
        self._transitions = task.get('transitions', np.full((num_states,
                                                             num_actions, num_states), 1.0 / num_states,
                                                            dtype=np.float32))
        self._rewards_mean = task.get('rewards_mean', np.zeros((num_states,
                                                                num_actions), dtype=np.float32))
        
        self._state = 0
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def sample_tasks(self, num_tasks):
        transitions = self.np_random.dirichlet(np.ones(self.num_states),
                                               size=(num_tasks, self.num_states, self.num_actions))
        rewards_mean = self.np_random.normal(1.0, 1.0,
                                             size=(num_tasks, self.num_states, self.num_actions))
        tasks = [{'transitions': transition, 'rewards_mean': reward_mean}
                 for (transition, reward_mean) in zip(transitions, rewards_mean)]
        
        return tasks

    def reset_task(self, task):
        self._task = task
        self._transitions = task['transitions']
        self._rewards_mean = task['rewards_mean']
        #print('Env Setup')
        #print(self._transitions)
        #print(self._rewards_mean)

    def reset(self):
        # From [1]: "an episode always starts on the first state"
        self._state = 0
        observation = np.zeros(self.num_states, dtype=np.float32)
        observation[self._state] = 1.0

        return observation

    def step(self, action):
        assert self.action_space.contains(action)
        mean = self._rewards_mean[self._state, action]
        reward = self.np_random.normal(mean, 1.0)

        self._state = self.np_random.choice(self.num_states,
                                            p=self._transitions[self._state, action])
        observation = np.zeros(self.num_states, dtype=np.float32)
        observation[self._state] = 1.0

        return observation, reward, False, self._task
