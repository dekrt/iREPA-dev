OUTPUT_DIR="/lpai/output/models/iREPA/eval/JiT_eval_irepa"
CKPT_DIR="/lpai/models/repa/jit-dinov3-vit-b16-irepa/jit-dinov3-vit-b16-irepa/checkpoint-80.pth"

mkdir -p /root/.cache/torch/hub/checkpoints/
cp /lpai/volumes/so-volume-bd-ga/lhp/pt_inception-2015-12-05-6726825d.pth /root/.cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth

CUDA_VISIBLE_DEVICES=2,3,4,6 \
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \
    --master_port 29600 \
    main_jit.py \
    --model JiT-B/16 \
    --img_size 256 --noise_scale 1.0 \
    --gen_bsz 1024 --num_images 50000 --cfg 1.0 \
    --output_dir ${OUTPUT_DIR} \
    --resume ${CKPT_DIR} \
    --report_to tensorboard \
    --evaluate_gen
