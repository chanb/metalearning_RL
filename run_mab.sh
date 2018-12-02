# python rl2_train.py --out_file test.pt --batch_size 10000 --num_tasks 1000 --mini_batch_size 256 --num_traj 100 --tau 0.3 --gamma 0.99 --ppo_epochs 10
# python rl2_eval.py --algo ppo --eval_tasks ./experiments/bandit_5_1000.pkl --out_file result.pkl --eval_model test.pt --num_traj 100
# python read_result.py --file result.pkl --task bandit --out_file exp1


python rl2_train.py --out_file test.pt --batch_size 15 --num_tasks 6 --mini_batch_size 10 --num_traj 6 --tau 0.3 --gamma 0.99 --learning_rate 0.05 --ppo_epochs 3 --clip_param 0.2 --traj_len 10 --task mdp