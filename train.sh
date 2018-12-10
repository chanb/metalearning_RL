#!/usr/bin/env bash

num_tasks=( 25000 2500 500 )
num_batches=( 500 50 10 ) #num_tasks divided by meta batch size which is 50
arms=( 5 )
trajs=( 10 100 500 )

i=0
while [  $i -lt ${#trajs[@]} ]; do
    for arm in ${arms[@]}; do
        echo arm is $arm
        echo fast_batch_size is  ${trajs[$i]}
        echo num_batches is ${num_batches[$i]}
        python maml.py --task bandit --num_actions $arm --fast-batch-size ${trajs[$i]} --num-batches ${num_batches[$i]} --meta-batch-size 50
    done
    let i=i+1
done

trajs=( 10 )
num_tasks=2500
num_batches=50 #num_tasks divided by meta batch size which is 50
for traj in ${trajs[@]}; do
    echo mdp
    python maml.py --task mdp --fast-batch-size $traj --num-batches $num_batches --meta-batch-size 50
done