python rl2_train.py --out_file test.pt --batch_size 10000 --num_tasks 2500 --mini_batch_size 256 --num_traj 10 --tau 0.3 --gamma 0.99
python rl2_eval.py --algo ppo --eval_tasks ./experiments/bandit_5_1000.pkl --out_file result.pkl --eval_model test.pt --num_traj 10
python read_result.py --file result.pkl --task bandit --out_file exp1


# python rl2_train.py --out_file test.pt --batch_size 20 --num_tasks 1 --mini_batch_size 10 --num_traj 1000 --tau 0.3 --gamma 0.99