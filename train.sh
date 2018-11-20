#!/usr/bin/env bash

num_tasks=100
arms=( 5 10 50 )
trajs=( 10 100 500 1000 )

for traj in ${trajs[@]}; do
    for arm in ${arms[@]}; do
        echo $arm
        echo $traj
        python rl2.py --learning_rate 0.001 --algo reinforce --task bandit --num_actions $arm --max_traj_len 1 --max_num_traj $traj --num_tasks $num_tasks
        python snail.py --learning_rate 0.001 --algo reinforce --task bandit --num_actions $arm --max_traj_len 1 --max_num_traj $traj --num_tasks $num_tasks
        python maml.py --env-name Bandit-K$arm-v0 --fast-batch-size $traj --num-batches $num_tasks
    done
done

trajs=( 10 25 50 75 100 )
for traj in ${trajs[@]}; do
    echo $traj
    python rl2.py --num_tasks $num_tasks --max_num_traj 10 --learning_rate 0.001 --algo reinforce --max_traj_len 10 --task mdp
    python snail.py --num_tasks $num_tasks --max_num_traj 10 --learning_rate 0.001 --algo reinforce --max_traj_len 10 --task mdp
    python maml.py --env-name TabularMDP-v0 --fast-batch-size $traj --num-batches $num_tasks
done