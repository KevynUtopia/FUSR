#!/bin/bash
# This is your shell script

# Define your command with complicated arguments
# Execute the command
GPU_id=0
CUDA_VISIBLE_DEVICES=$GPU_id python train.py --n_epochs 50 \
--l2h_hb 3 --l2h_lb 17 --h2l_hb 3 --h2l_lb 17 --dis_iter 1 --hfb 1.0 --input_D_nc 1 --loss_weight 10 \
--ffl_h --ffl_l --tensorboard


