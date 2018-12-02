python rl2_train.py --out_file test.pt --batch_size 100 --num_tasks 1000 --mini_batch_size 25 --num_traj 10 --tau 0.3 --gamma 0.99
python rl2_eval.py --algo ppo --eval_tasks ./experiments/bandit_5_1000.pkl --out_file result.pkl --eval_model test.pt --num_traj 10
python read_result.py --file result.pkl --task bandit --out_file exp1


# python rl2_train.py --out_file test.pt --batch_size 20 --num_tasks 1 --mini_batch_size 10 --num_traj 100 --tau 0.3 --gamma 0.99 --learning_rate 0.05 --ppo_epochs 3 --clip_param 0.2