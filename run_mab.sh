tasks=100
lr=0.0002
epochs=8
batch=300
minibatch=100
traj=100

eval_minibatch=2
eval_batchsize=5
eval_ppo_epochs=8
eval_tasks=100
# eval_lr=0.05
eval_lr=0.0002


# Random pulls
# python evaluate_model.py --algo random --num_tasks $eval_tasks --num_actions 5 --task bandit --eval_tasks ./experiments/bandit_5_$eval_tasks.pkl --outfile random_result.pkl --num_traj $traj

# python read_result.py --task bandit --file ./random_result.pkl --outfile ./rand_plot_$traj

# python rl2.py --algo ppo --ppo_epochs $epochs --task bandit --traj_len 1 --num_traj $traj --mini_batch_size $minibatch --learning_rate $lr --batch_size $batch --num_tasks $tasks --clip_param 0.2

python evaluate_model.py --num_tasks $eval_tasks --num_actions 5 --task bandit --eval_model ./saves/rl2/ppo_bandit_5_${traj}_SGD_lr${lr}_numtasks$tasks.pt --eval_tasks ./experiments/bandit_5_$eval_tasks.pkl --outfile sgd_result.pkl --ppo_epochs $eval_ppo_epochs --traj_len 1 --num_traj $traj --mini_batch_size $eval_minibatch --learning_rate $eval_lr --batch_size $eval_batchsize --clip_param 0.2

python read_result.py --task bandit --file ./sgd_result.pkl --outfile ./sgd_plot_$traj

python compare_actions_env.py --res_file ./sgd_result.pkl --env_file ./experiments/bandit_5_$eval_tasks.pkl

#Best: batch = 300, minibatch = 100, epochs = 8, lr = 0.0002
#Tried: batch = 200, minibatch = 50, epochs = 8, lr = 0.0002
#Tried: batch = 300, minibatch = 50, epochs = 10, lr = 0.0002

#Trying (Currently good): batch = 300, minibatch = 100, epochs = 8, lr = 0.0002