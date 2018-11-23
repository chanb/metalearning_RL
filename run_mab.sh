tasks=100
lr=0.0002
epochs=4
batch=10
minibatch=5

python rl2.py --algo ppo --ppo_epochs $epochs --task bandit --traj_len 1 --num_traj 100 --mini_batch_size $minibatch --learning_rate $lr --batch_size $batch --num_tasks $tasks --clip_param 0.1

python evaluate_model.py --num_tasks $tasks --num_actions 5 --task bandit --eval_model ./saves/rl2/ppo_bandit_5_100_SGD_lr${lr}_numtasks$tasks.pt --eval_tasks ./experiments/bandit_5_$tasks.pkl --outfile sgd_result.pkl --ppo_epochs $epochs --task bandit --traj_len 1 --num_traj 100 --mini_batch_size $minibatch --learning_rate $lr --batch_size $batch --clip_param 0.1


python read_result.py --task bandit --file ./sgd_result.pkl --outfile ./sgd_plot_$tasks

python compare_actions_env.py --res_file ./sgd_result.pkl --env_file ./experiments/bandit_5_$tasks.pkl
