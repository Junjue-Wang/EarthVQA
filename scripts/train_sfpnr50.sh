#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:`pwd`
config_path='sfpnr50'
model_dir='./log/sfpnr50'
NUM_GPUS=1

python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port $RANDOM train_lovedav2_seg.py \
    --config_path=${config_path} \
    --model_dir=${model_dir} \
