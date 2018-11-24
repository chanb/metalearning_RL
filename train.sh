#!/usr/bin/env bash

python maml.py --env-name 2DNavigation-v0
python test.py --policy-file ./saves/maml/policy.pt # test trained maml
python test.py # test untrained maml: equivalent to just training a normal Gaussian MLP