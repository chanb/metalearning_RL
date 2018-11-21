#!/usr/bin/env bash
# Find the best learning rate

arm=5
traj=10
lrs=( 0.0001 0.0005 0.001 0.005 0.01 )
num_tasks=( 50 100 200 )
tasks=( bandit mdp )

for task in ${tasks[@]}; do
    for num_task in ${num_tasks[@]}; do
        for lr in ${lrs[@]}; do
            python rl2.py --learning_rate $lr --algo reinforce --task $task --num_actions $arm --max_num_traj $traj --num_tasks $num_task
            python snail.py --learning_rate $lr --algo reinforce --task $task --num_actions $arm --max_num_traj $traj --num_tasks $num_task
        done
        python maml.py -task $task --num_actions $arm --fast-batch-size $traj --num-batches $num_tasks
    done
done