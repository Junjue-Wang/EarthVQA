#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:`pwd`
config_path='sfpnr50'
ckpt_path='./log/sfpnr50.pth'
save_dir='./log/sfpnr50/test_features'

python ./predict_seg.py \
    --config_path=${config_path} \
    --ckpt_path=${ckpt_path} \
    --save_dir=${save_dir}
