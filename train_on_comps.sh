#!/usr/bin/env bash

arms=5
task=bandit
num_tasks=10
num_traj=10
traj_len=1

tau=0.3
gamma=0.99
ppo_epoch=5
learning_rate=3e-4
clip_param=0.1
num_workers=3

batchsize=10
minibatchsize=5
metalearn_epoch=100

models=( gru )

# For each of the model, run MAB
for model in ${models[@]}; do
    policy=train_${model}_${arms}${task}_${num_traj}traj_${num_tasks}tasks
    mkdir -p $policy

    python -W ignore rl2_train.py --task $task --model_type $model --out_file $policy.pt --batch_size $batchsize --num_tasks $num_tasks --mini_batch_size $minibatchsize --num_traj $num_traj --tau $tau --gamma $gamma --ppo_epochs $ppo_epoch --learning_rate $learning_rate --clip_param $clip_param --num_workers $num_workers --num_actions $arms --traj_len $traj_len --metalearn_epoch $metalearn_epoch

    mv ./tmp/*_$policy.pt $policy
    mv $policy.pt $policy
done
