#!/bin/bash

source /root/anaconda3/etc/profile.d/conda.sh
conda activate dit

cd /lpai/volumes/so-volume-bd-ga/lhp/code/zxk/iREPA-dev/jit

mkdir -p /root/.cache/torch/hub/checkpoints/
cp /lpai/volumes/so-volume-bd-ga/lhp/pt_inception-2015-12-05-6726825d.pth /root/.cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth

MODELS=(
    "jit-dinov3-vit-b16-irepa-cosine-wo-spnorm"
    "jit-dinov3-vit-b16-irepa-semantic-nce-wo-spnorm"
)

LOG_FILE="/lpai/output/models/jit_irepa_wo_spnorm/eval_results_all.txt"
> $LOG_FILE

for MODEL_NAME in "${MODELS[@]}"; do
    BASE_CKPT_DIR="/lpai/input/models/repa/${MODEL_NAME}/${MODEL_NAME}"
    OUTPUT_DIR="/lpai/output/models/${MODEL_NAME}_eval"

    echo "Evaluating: ${MODEL_NAME}" | tee -a $LOG_FILE

#    for STEP in {20..200..20}; do
    for STEP in 40 60 80; do
        CKPT_NAME="checkpoint-${STEP}.pth"
        CKPT_PATH="${BASE_CKPT_DIR}/${CKPT_NAME}"

        if [ ! -f "$CKPT_PATH" ]; then
            echo "[警告] 文件未找到: $CKPT_PATH, 跳过..." | tee -a $LOG_FILE
            continue
        fi

        echo "-> ${CKPT_NAME}" | tee -a $LOG_FILE
        CUDA_VISIBLE_DEVICES=4,5,6,7 \
        torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \
            --master_port 29900 \
            main_jit.py \
            --model JiT-B/16 \
            --img_size 256 --noise_scale 1.0 \
            --gen_bsz 2048 --num_images 50000 --cfg 3.0 --interval_min 0.1 --interval_max 1.0 \
            --output_dir ${OUTPUT_DIR} \
            --resume ${CKPT_PATH} \
            --report_to tensorboard \
            --evaluate_gen 2>&1 | tee -a $LOG_FILE
    done
done

sleep 1d;