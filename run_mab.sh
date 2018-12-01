python rl2_train.py --out_file test.pt --batch_size 1000 --num_tasks 1000 --mini_batch_size 256 --num_traj 10
python rl2_eval.py --algo ppo --eval_tasks ./experiments/bandit_5_1000.pkl --out_file result.pkl --eval_model test.pt --num_traj 10
python read_result.py --file result.pkl --task bandit --out_file exp1