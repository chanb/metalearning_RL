# SNAIL_Pytorch

This project follows RL^2 and SNAIL paper and attempts to reproduce the results presented.
`rl2_train.py` trains a model, given the arguments:
- `num_workers (int)`: (Default: `1`) This spawns specified amount of workers to perform parallel sampling (# of actors in PPO paper)
- `model_type (str)`: (Default: `gru`) This chooses either `gru` (from RL^2) or `snail` as the model to train
- `metalearn_epochs (int)` (Default: `300`) This is the number of iterations to train the model. After each iteration, a snapshot is taken for plotting the learning curve
- `task (str)`: (Default: `mdp`) This supports either `mdp` (tabular MDP) or `bandit` (multi-armed bandit)
- `non_linearity (str)`: (Default: `none`) This is the non linearity function following the last output layer of the value network. This supports `none`, `tanh`, `relu`, and `sigmoid`
- `learning_rate (float)`: (Default: `3e-4`) This is the learning rate for the Adam optimizer
- `gamma (float)`: (Default: `0.99`) This is the discount factor
- `num_actions (int)`: (Default: `5`) This is the number of actions in the action space for the given task. This should only be tweaked for multi-armed bandit (unless new environment is provided)
- `num_tasks (int)`: (Default: `1000`) This specifies the number of tasks to learn from
- `num_traj (int)`: (Default: `10`) This specifies the number of trajectories to interact with given a task
- `traj_len (int)`: (Default: `1`) This specifies the length of the trajectory to sample from
- `tau (float)`: (Default: `0.95`) This is the GAE lambda parameter
- `mini_batch_size (int)`: (Default: `256`) This is the minibatch size `M` from PPO paper. This needs to be less than or equal to `batch_size`
- `batch_size (int)`: (Default: `10000`) This is the batch size `T` from the PPO paper. This essentially means we sample `T` actions before a PPO update
- `ppo_epochs (int)`: (Default: `5`) This is the PPO epoch `K`.
- `clip_param (float)`: (Default: `0.1`) This is the clipping factor `epsilon` from the PPO paper
- `vf_coef (float)`: (Default: `0.5`) This is the value function loss coefficient for the loss function
- `ent_coef (float)`: (Default: `0.01`) This is the entropy coefficient for the loss function
- `max_grad_norm (float)`: (Default: `0.9`) This clips the gradient if its norm exceeds the maximum norm allowed
- `target_kl (float)`: (Default: `0.01`) This is the mean KL that early stops the update if the KL divergence between old and new polcies is too high
- `out_file (str)`: This is the name of the model file to be output

## Installation:
You should be able to simply run `pip install` on the `requirements.txt`:  
`$ pip install -r requirements.txt`

## Sample Commands:
```
$ python rl2_train.py --out_file test_bandit.pt --batch_size 100 --num_tasks 100 --mini_batch_size 20 --num_traj 50 --tau 0.3 --gamma 0.99 --ppo_epochs 7 --learning_rate 0.005 --clip_param 0.2
$ python rl2_train.py --out_file test_10armed_bandit.pt --batch_size 100 --num_tasks 100 --mini_batch_size 20 --num_traj 50 --tau 0.3 --gamma 0.99 --ppo_epochs 7 --learning_rate 0.005 --clip_param 0.2 --num_actions 10
$ python rl2_train.py --out_file test_mdp_snail.pt --batch_size 100 --num_tasks 100 --mini_batch_size 20 --num_traj 50 --tau 0.3 --gamma 0.99 --ppo_epochs 7 --learning_rate 0.005 --clip_param 0.2 --model_type snail --metalearn_epochs 1500
```

## Experiment Status:
- 5 Armed Bandit:
  - 10 Trajectories, 25000 Tasks (compsbk3)
    - GRU: Running
    - SNAIL: Running  
  - 100 Trajectories, 2500 Tasks (compsgpu2)
    - GRU: Running
    - SNAIL: Running
  - 500 Trajectories, 500 Tasks (angeline's vector machine)
    - GRU: Running
    - SNAIL: Running
- **10 Armed Bandit (To be run)**:
- **50 Armed Bandit (To be run)**:
- Tabular MDP:  
  - 10 Trajectories, 2500 Tasks (compsbk3)
    - GRU: Running
    - SNAIL: Running  
  - 25 Trajectories, 1000 Tasks
    - GRU: Not Started
    - SNAIL: Not Started  
  - 50 Trajectories, 500 Tasks
    - GRU: Not Started
    - SNAIL: Not Started  
  - 75 Trajectories, 333 Tasks
    - GRU: Not Started
    - SNAIL: Not Started  
  - 100 Trajectories, 250 Tasks
    - GRU: Not Started
    - SNAIL: Not Started

## References:
https://github.com/higgsfield/RL-Adventure-2/blob/master/3.ppo.ipynb  
https://arxiv.org/abs/1707.06347  
https://github.com/VashishtMadhavan/rl2  
https://github.com/noahgolmant/RL-squared  
https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py  
https://medium.com/@sanketgujar95/trust-region-policy-optimization-trpo-and-proximal-policy-optimization-ppo-e6e7075f39ed  
https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be

### Reference codes: MAML-PyTorch, RL-Adventure, pytorch-ppo-acktr-a2c, openAI baseline, openAI spinningup
