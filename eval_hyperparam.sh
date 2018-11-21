#!/usr/bin/env bash
# Find the best learning rate

arm=5
traj=10
lrs=( 0.0001 0.0005 0.001 0.005 0.01 )
num_tasks=( 50 100 200 )
tasks=( bandit mdp )

for task in ${tasks[@]}; do
    python generate_experiments.py --num_tasks 100 --num_actions $arm --task $task
    for num_task in ${num_tasks[@]}; do
        for lr in ${lrs[@]}; do
           model=reinforce_$task"_"$arm"_"$traj"_"adam"_"lr$lr"_"numtasks$num_task
           python evaluate_model.py --num_tasks 100 --num_actions $arm --task $task \
                    --eval_model ./saves/rl2/$model.pt
                    --eval_tasks ./experiments/"$task"_"$action"_100.pkl
                    --out_file ./logs_eval/rl2/$model

           python evaluate_model.py --num_tasks 100 --num_actions $arm --task $task \
                    --eval_model ./saves/snail/$model.pt
                    --eval_tasks ./experiments/"$task"_"$action"_100.pkl
                    --out_file ./logs_eval/snail/$model
        done
        model=reinforce_$task"_"$arm"_"$traj"_"numtasks$num_task
        python evaluate_model.py --num_tasks 100 --num_actions $arm --task $task \
                    --eval_model ./saves/maml/$model.pt
                    --eval_tasks ./experiments/"$task"_"$action"_100.pkl
                    --out_file ./logs_eval/maml/$model
    done
done