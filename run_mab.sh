python rl2_train.py --out_file test.pt --batch_size 100 --num_tasks 100 --mini_batch_size 25 --num_traj 100
python rl2_eval.py --algo ppo --eval_tasks ./experiments/bandit_5_100.pkl --out_file result.pkl --eval_model test.pt --num_traj 100
python read_result.py read_result.py -file result.pkl --task bandit --outfile exp_1