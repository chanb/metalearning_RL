import envs
import gym
import numpy as np
import torch
import json

from metalearner import MetaLearner
from policies import CategoricalMLPPolicy
from baseline import LinearFeatureBaseline
from sampler import BatchSampler
from utils.torch_utils import (weighted_mean, detach_distribution,
                                       weighted_normalize)

from tensorboardX import SummaryWriter

def total_rewards(episodes_rewards, aggregation=torch.mean):
    rewards = torch.mean(torch.stack([aggregation(torch.sum(rewards, dim=0))
        for rewards in episodes_rewards], dim=0))
    return rewards.item()

def inner_loss(episodes, baseline, policy):
    """Compute the inner loss for the one-step gradient update. The inner 
    loss is REINFORCE with baseline [2], computed on advantages estimated 
    with Generalized Advantage Estimation (GAE, [3]).
    """
    values = baseline(episodes)
    advantages = episodes.gae(values, tau=args.tau)
    advantages = weighted_normalize(advantages, weights=episodes.mask)

    pi = policy(episodes.observations)
    log_probs = pi.log_prob(episodes.actions)
    if log_probs.dim() > 2:
        log_probs = torch.sum(log_probs, dim=2)
    loss = -weighted_mean(log_probs * advantages, dim=0,
        weights=episodes.mask)

    return loss

def main(args):
    writer = SummaryWriter('./logs/{0}'.format(args.output_folder))
    save_folder = './saves/{0}'.format(args.output_folder)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    with open(os.path.join(save_folder, 'config.json'), 'w') as f:
        config = {k: v for (k, v) in vars(args).items() if k != 'device'}
        config.update(device=args.device.type)
        json.dump(config, f, indent=2)

    sampler = BatchSampler(args.env_name, batch_size=args.task_batch_size, num_workers=1)
    baseline = LinearFeatureBaseline(int(np.prod(sampler.envs.observation_space.shape)))

    # TODO: Allow other policy (GRU, SNAIL, etc)
    policy = CategoricalMLPPolicy(int(np.prod(sampler.envs.observation_space.shape)), sampler.envs.action_space.n, hidden_sizes=(args.hidden_size,) * args.num_layers)

    # Sample next batch of tasks (in parallel)
    tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)
    
    # Sample trajectories for every task
    for task in tasks:
        print('====================================================================================')
        print(task)
        for _ in range(args.num_trajectories):
            trajectory = sampler.sample_trajectory(policy, task, args.gamma, args.device)

            print("actions")
            print(trajectory.actions)
            print("rewards")
            print(trajectory.rewards)

            baseline.fit(trajectory)
            loss = inner_loss(trajectory, baseline, policy)
            policy.update_params(loss, step_size=args.task_lr, first_order=args.first_order)
    
    # Save policy network
    if (args.save):
        with open(os.path.join(save_folder, 'policy-0.pt'), 'wb') as f:
            torch.save(policy.state_dict(), f)


if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='Reinforcement learning with '
        'Model-Agnostic Meta-Learning (MAML)')

    # General
    parser.add_argument('--env-name', type=str,
        help='name of the environment')
    parser.add_argument('--gamma', type=float, default=0.95,
        help='value of the discount factor gamma')
    parser.add_argument('--tau', type=float, default=1.0,
        help='value of the discount factor for GAE')
    parser.add_argument('--first-order', action='store_true',
        help='use the first-order approximation of MAML')

    # Policy network (relu activation function)
    parser.add_argument('--hidden-size', type=int, default=100,
        help='number of hidden units per layer')
    parser.add_argument('--num-layers', type=int, default=2,
        help='number of hidden layers')

    # Task-specific
    parser.add_argument('--task-batch-size', type=int, default=1,
        help='batch size for each individual task')
    parser.add_argument('--task-lr', type=float, default=0.5,
        help='learning rate for the 1-step gradient update')

    # Optimization
    parser.add_argument('--num-trajectories', type=int, default=5,
        help='number of trajectories')
    parser.add_argument('--meta-batch-size', type=int, default=5,
        help='number of tasks per batch')
    parser.add_argument('--max-kl', type=float, default=1e-2,
        help='maximum value for the KL constraint in TRPO')
    parser.add_argument('--cg-iters', type=int, default=10,
        help='number of iterations of conjugate gradient')
    parser.add_argument('--cg-damping', type=float, default=1e-5,
        help='damping in conjugate gradient')
    parser.add_argument('--ls-max-steps', type=int, default=15,
        help='maximum number of iterations for line search')
    parser.add_argument('--ls-backtrack-ratio', type=float, default=0.8,
        help='maximum number of iterations for line search')

    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='maml',
        help='name of the output folder')
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling')
    parser.add_argument('--device', type=str, default='cpu',
        help='set the device (cpu or cuda)')

    parser.add_argument('--save',type=bool, default=False, help='Save the weights for policy')

    args = parser.parse_args()

    # Create logs and saves folder if they don't exist
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./saves'):
        os.makedirs('./saves')
    # Device
    args.device = torch.device(args.device
        if torch.cuda.is_available() else 'cpu')
    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])

    main(args)
