#!/usr/bin/env bash

arms=( 5 10 50 )
tasks=( bandit mdp )
num_tasks=100

algos=( rl2 snail maml )
trajs=( 10 100 500 1000 )

for task in ${tasks[@]}; do
    if [ $task == "bandit" ]; then
        for arm in ${arms[@]}; do
            python generate_experiments.py --num_tasks $num_tasks --num_actions $arm --task $task
        done
    else
        python generate_experiments.py --num_tasks $num_tasks --num_actions 5 --task $task
    fi
done

for algo in ${algos[@]}; do
    python evaluate_model.py --num_tasks $num_tasks --num_actions $arm --task $task --eval_model --eval_tasks ./experiments/"$task"_"$num_actions"_"$num_tasks"
done