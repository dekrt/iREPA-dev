#!/usr/bin/env bash
# Script: train_jit_wo_repa.sh

set -euo pipefail

source /root/anaconda3/etc/profile.d/conda.sh
conda activate dit

cd /lpai/volumes/so-volume-bd-ga/lhp/code/zxk/iREPA-dev/jit

mkdir -p /root/.cache/torch/hub/checkpoints/
cp /lpai/volumes/so-volume-bd-ga/lhp/pt_inception-2015-12-05-6726825d.pth /root/.cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth

CUDA_VISIBLE_DEVICES=4,6 \
torchrun --nproc_per_node=2 main_jit.py \
  --model JiT-B/16 \
  --enc_type="dinov3-vit-b16" \
  --encoder_depth=4 \
  --data_path=../data \
  --epochs 200 \
  --output_dir="/lpai/output/models/jit-dinov3-vit-b16-jit-wo-repa" \
  --report_to tensorboard \
  --batch_size 16

sleep 1d
