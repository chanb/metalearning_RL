#!/usr/bin/env bash
# Find the best learning rate

arm=5
traj=10
lrs=( 0.001 )
num_tasks=( 50 100 200 )
tasks=( bandit mdp )
mkdir -p ./logs_eval/rl2
mkdir -p ./logs_eval/maml
mkdir -p ./logs_eval/snail

for task in ${tasks[@]}; do
    python generate_experiments.py --num_tasks 100 --num_actions $arm --task $task
    for num_task in ${num_tasks[@]}; do
        for lr in ${lrs[@]}; do
           model=reinforce_$task"_"$arm"_"$traj"_"adam"_"lr$lr"_"numtasks$num_task
           python evaluate_model.py --num_tasks 100 --num_actions $arm --task $task \
                    --eval_model ./saves/rl2/$model.pt \
                    --eval_tasks ./experiments/"$task"_"$arm"_100.pkl \
                    --outfile ./logs_eval/rl2/$model.pkl

           python read_result.py --task $task \
                    --file ./logs_eval/rl2/$model.pkl \
                    --outfile ./plots/rl2/$model

           python evaluate_model.py --num_tasks 100 --num_actions $arm --task $task \
                    --eval_model ./saves/snail/$model.pt \
                    --eval_tasks ./experiments/"$task"_"$arm"_100.pkl \
                    --outfile ./logs_eval/snail/$model.pkl

            python read_result.py --task $task \
                    --file ./logs_eval/snail/$model.pkl \
                    --outfile ./plots/snail/$model
        done
    done
done
