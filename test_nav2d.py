import numpy as np
import torch

from helper.envs.navigation import Navigation2DEnv
from helper.policies.normal_mlp import NormalMLPPolicy
from helper.baseline import LinearFeatureBaseline
from helper.sampler import BatchSampler
from helper.metalearner import MetaLearner

ITR = 40
# from zhanpenghe: https://github.com/tristandeleu/pytorch-maml-rl/issues/15
# torch.manual_seed(7)

META_POLICY_PATH = "./saves/maml/policy-{}.pt".format(ITR)
#BASELINE_PATH = "/saves/maml/baseline-{}.pt".format(ITR)
BASELINE_PATH = None

TEST_TASKS = [
    (5., 5.)
]


def load_meta_learner_params(policy_path, baseline_path, env):
    policy_params = torch.load(policy_path)

    policy = NormalMLPPolicy(
        int(np.prod(env.observation_space.shape)),
        int(np.prod(env.action_space.shape)),
        hidden_sizes=(100, 100))  # We should actually get this from config
    policy.load_state_dict(policy_params)

    baseline = LinearFeatureBaseline(int(np.prod(env.observation_space.shape)))
    if baseline_path:
        baseline_params = torch.load(baseline_path)
        baseline.load_state_dict(baseline_params)

    return policy, baseline


def evaluate(env, task, policy, max_path_length=100):
    cum_reward = 0
    t = 0
    env.reset_task(task)
    obs = env.reset()
    for _ in range(max_path_length):
        #env.render()
        obs_tensor = torch.from_numpy(obs).to(device='cpu').type(torch.FloatTensor)
        action_tensor = policy(obs_tensor, params=None).sample()
        action = action_tensor.cpu().numpy()
        obs, rew, done, _ = env.step(action)
        cum_reward += rew
        t += 1
        if done:
            break

    print("========EVAL RESULTS=======")
    print("Return: {}, Timesteps:{}".format(cum_reward, t))
    print("===========================")


def main():
    env = Navigation2DEnv()
    policy, baseline = load_meta_learner_params(META_POLICY_PATH, BASELINE_PATH, env)
    sampler = BatchSampler(env_name="2DNavigation-v0", batch_size=20, num_workers=2)
    learner = MetaLearner(sampler, policy, baseline)
    tasks = sampler.sample_tasks(num_tasks=400)
    for task in tasks:
        env.reset_task(task)

        # Sample a batch of transitions
        sampler.reset_task(task)
        episodes = sampler.sample(policy)
        new_params = learner.adapt(episodes)
        policy.load_state_dict(new_params)
        evaluate(env, task, policy)


if __name__ == '__main__':
    main()