tasks=100

python rl2.py --algo ppo --ppo_epochs 4 --task bandit --traj_len 1 --num_traj 100 --mini_batch_size 5 --learning_rate 0.0002 --batch_size 10 --num_tasks $tasks --clip_param 0.1

python evaluate_model.py --num_tasks $tasks --num_actions 5 --task bandit --eval_model ./saves/rl2/ppo_bandit_5_100_SGD_lr0.0002_numtasks$tasks.pt --eval_tasks ./experiments/bandit_5_$tasks.pkl --outfile sgd_result.pkl --ppo_epochs 4 --task bandit --traj_len 1 --num_traj 100 --mini_batch_size 5 --learning_rate 0.0002 --batch_size 10 --clip_param 0.1


python read_result.py --task bandit --file ./sgd_result.pkl --outfile ./sgd_plot_$tasks

python compare_actions_env.py --res_file ./sgd_result.pkl --env_file ./experiments/bandit_5_$tasks.pkl
