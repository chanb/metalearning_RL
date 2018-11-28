import numpy as np
import torch

from helper.envs.navigation import Navigation2DEnv
from helper.policies.normal_mlp import NormalMLPPolicy
from helper.baseline import LinearFeatureBaseline
from helper.sampler import BatchSampler
from helper.metalearner import MetaLearner

import matplotlib as mpl
import matplotlib.pyplot as plt
#mpl.use('TkAgg')

# from zhanpenghe: https://github.com/tristandeleu/pytorch-maml-rl/issues/15
# torch.manual_seed(7)


def load_meta_learner_params(env, args):
    if args.policy_file:
        policy = torch.load(args.policy_file)
    else:
        policy = NormalMLPPolicy(
            int(np.prod(env.observation_space.shape)),
            int(np.prod(env.action_space.shape)),
            hidden_sizes=(args.hidden_size,) * args.num_layers)

    if args.baseline_file:
        baseline = torch.load(args.baseline_file)
    else:
        baseline = LinearFeatureBaseline(int(np.prod(env.observation_space.shape)))
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
    return cum_reward, t


def main(args):
    env = Navigation2DEnv()
    sampler = BatchSampler(env_name=args.env_name, batch_size=args.fast_batch_size, num_workers=args.num_workers)
    tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)
    means = [[] for _ in range(args.num_grad_step)]
    timesteps = [[] for _ in range(args.num_grad_step)]
    for task in tasks:
        env.reset_task(task)
        policy, baseline = load_meta_learner_params(env, args)
        metalearner = MetaLearner(sampler, policy, baseline) #use metalearner architecture but not actually meta learning for the test
        metalearner.fast_lr = args.fast_lr
        # Sample a batch of transitions
        for grad in range(args.num_grad_step):
            cum_rewards = []
            ts = []
            print("========GRAD STEP {}========".format(grad))
            for _ in range(args.test_iter):
                cum_reward, t = evaluate(env, task, policy)
                cum_rewards.append(cum_reward)
                ts.append(t)
            print("========EVAL RESULTS=======")
            print("Return: {} +- {}, Timesteps:{} +- {}".format(np.mean(cum_rewards), np.std(cum_rewards),
                                                                np.mean(ts), np.std(ts)))
            print("===========================")
            means[grad].append(np.mean(cum_rewards))
            timesteps[grad].append(np.mean(ts))

            sampler.reset_task(task)
            episodes = sampler.sample(policy)
            new_params = metalearner.adapt(episodes, first_order=args.first_order)
            policy.load_state_dict(new_params)

    # plotting
    means = np.array(means)
    timesteps=np.array(timesteps)
    plt.figure(0)
    plt.plot(range(args.num_grad_step), np.mean(means, axis=1))
    plt.xlabel('Number of Gradient Updates')
    plt.ylabel('Average Return')
    plt.title('Model Performance')
    plt.fill_between(range(args.num_grad_step), np.mean(means, axis=1) - np.std(means, axis=1),
                     np.mean(means, axis=1) + np.std(means, axis=1), color = 'blue', alpha=0.3, lw=0.001)
    plt.savefig('plots/{}-means.png'.format(args.outfile))

    plt.figure(1)
    plt.plot(range(args.num_grad_step), np.mean(timesteps, axis=1))
    plt.xlabel('Number of Gradient Updates')
    plt.ylabel('Average Timestep Taken')
    plt.title('Model Performance')
    plt.fill_between(range(args.num_grad_step), np.mean(timesteps, axis=1) - np.std(timesteps, axis=1),
                     np.mean(timesteps, axis=1) + np.std(timesteps, axis=1), color='blue', alpha=0.3, lw=0.001)
    plt.savefig('plots/{}-timesteps.png'.format(args.outfile))


if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='Test MAML on 2DNavigation')

    # General
    parser.add_argument('--env-name', type=str, default='2DNavigation-v0',
                        help='name of the environment')
    parser.add_argument('--first-order', action='store_true',
                        help='use the first-order approximation of MAML')

    # Policy network (relu activation function)
    parser.add_argument('--hidden-size', type=int, default=100,
                        help='number of hidden units per layer')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='number of hidden layers')

    # Task-specific
    parser.add_argument('--fast-batch-size', type=int, default=20,
                        help='batch size for each individual task')

    # Optimization
    parser.add_argument('--num-batches', type=int, default=200,
                        help='number of batches')
    parser.add_argument('--meta-batch-size', type=int, default=40,
                        help='number of tasks per batch')

    # Test arguments
    parser.add_argument('--test-iter', type=int, default=10,
                        help='evaluate the same task at the same gradient update test-iter times')
    parser.add_argument('--num-grad-step', type=int, default=5,
                        help='Number of gradient steps to try')
    parser.add_argument('--fast-lr', type=int, default=0.1,
                        help='Number of gradient steps to try')

    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='maml',
                        help='name of the output folder')
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
                        help='number of workers for trajectories sampling')
    parser.add_argument('--device', type=str, default='cpu',
                        help='set the device (cpu or cuda)')

    parser.add_argument('--policy-file', type=str,
                        help='Name of policy file')
    parser.add_argument('--baseline-file', type=str,
                        help='Name of baseline file') #optional
    parser.add_argument('--outfile', type=str,
                        help='Name of output file')

    args = parser.parse_args()

    # Create logs and saves folder if they don't exist
    if not os.path.exists('./plots'):
        os.makedirs('./plots')
    # Device
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])

    main(args)