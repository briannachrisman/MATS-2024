#!/bin/bash

python -u train.py --l1_coefficients=[3] --hook_point_head_index=$1 --total_training_steps=300000
python -u train.py --l1_coefficients=[3] --hook_point_head_index=$2 --total_training_steps=300000
python -u train.py --l1_coefficients=[3] --hook_point_head_index=$3 --total_training_steps=300000
python -u train.py --l1_coefficients=[3] --hook_point_head_index=$4 --total_training_steps=300000

