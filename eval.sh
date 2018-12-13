#!/usr/bin/env bash

trajs=( 10 100 500 )
for traj in ${trajs[@]}; do
    python maml_test.py --test-iter $traj --task bandit --outfile maml_bandit_5arm_$traj"traj" --policy-file ./saves/maml/reinforce_bandit_5_$traj.pt
    python maml_test.py --test-iter $traj --task bandit --outfile nomaml_bandit_5arm_$traj"traj"
done

python maml_test.py --test-iter 10 --task mdp --outfile maml_mdp_10traj --policy-file ./saves/maml/reinforce_mdp_5_10.pt
python maml_test.py --test-iter 10 --task mdp --outfile nomaml_mdp_10traj