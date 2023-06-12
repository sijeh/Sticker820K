#!/usr/bin/env

# Guide:
# This script supports distributed training on multi-gpu workers (as well as single-worker training).
# Please set the options below according to the comments.
# For multi-gpu workers training, these options should be manually set for each worker.
# After setting the options, please run the script on each worker.
# Command: bash run_scripts/muge_finetune_vit-b-16_rbt-base.sh ${DATAPATH}

# # Number of GPUs per GPU worker
# GPUS_PER_NODE=8
# # Number of GPU workers, for single-worker training, please set to 1
# WORKER_CNT=1
# # The ip address of the rank-0 worker, for single-worker training, please set to localhost
# export MASTER_ADDR=localhost
# # The port for communication
# export MASTER_PORT=8514
# # The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
# export RANK=0

export MASTER_PORT=8514
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=eth1

export PYTHONPATH=${PYTHONPATH}:StickerCLIP/sticker_clip

# DATAPATH=${1}

# data options
train_data=lmdb_data/EmojiData


# restore options
resume=pretrained_checkpoint.pt # or specify your customed ckpt path to resume
reset_data_offset="--reset-data-offset"
reset_optimizer="--reset-optimizer"
# reset_optimizer=""

# output options
output_base_dir=StickerCLIP/work_dir/
# name=flickr30k_finetune_vit-b-16_roberta-base_bs128_8gpu
name="emoji_finetune_vit-b"
save_step_frequency=999999 # disable it
save_epoch_frequency=1
log_interval=10
report_training_batch_acc="--report-training-batch-acc"
# report_training_batch_acc=""

# training hyper-params
context_length=64
warmup=100
batch_size=64
valid_batch_size=64
precision="amp"
lr=2e-5
# lr=5e-7
wd=0.001
max_epochs=5 # or you can alternatively specify --max-steps
valid_step_interval=300
valid_epoch_interval=1
vision_model=ViT-B-16
text_model=RoBERTa-wwm-ext-base-chinese
use_augment="--use-augment"
# use_augment=""

python3 -m torch.distributed.launch --nproc_per_node=$HOST_GPU_NUM --nnodes=$HOST_NUM --node_rank=$INDEX \
    --master_addr=$CHIEF_IP --master_port=${MASTER_PORT} StickerCLIP/sticker_clip/training/main.py \
    --train-data=${train_data} \
    --val-data=${val_data} \
    --resume=${resume} \
    ${reset_data_offset} \
    ${reset_optimizer} \
    --logs=${output_base_dir} \
    --name=${name} \
    --save-step-frequency=${save_step_frequency} \
    --save-epoch-frequency=${save_epoch_frequency} \
    --log-interval=${log_interval} \
    ${report_training_batch_acc} \
    --context-length=${context_length} \
    --warmup=${warmup} \
    --batch-size=${batch_size} \
    --valid-batch-size=${valid_batch_size} \
    --valid-step-interval=${valid_step_interval} \
    --valid-epoch-interval=${valid_epoch_interval} \
    --lr=${lr} \
    --wd=${wd} \
    --max-epochs=${max_epochs} \
    --vision-model=${vision_model} \
    ${use_augment} \
    --text-model=${text_model} \
    --precision=${precision}