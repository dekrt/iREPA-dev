OUTPUT_DIR="/lpai/output/models/iREPA/eval/JiT_dynamic_coeff"
CKPT_DIR="/lpai/models/repa/jit-dinov3-vit-b16-irepa-v2/jit-dinov3-vit-b16-irepa/checkpoint-200.pth"

mkdir -p /root/.cache/torch/hub/checkpoints/
cp /lpai/volumes/so-volume-bd-ga/lhp/pt_inception-2015-12-05-6726825d.pth /root/.cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth

CUDA_VISIBLE_DEVICES=4,5,6,7 \
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \
    --master_port 29600 \
    main_jit.py \
    --model JiT-B/16 \
    --img_size 256 --noise_scale 1.0 \
    --gen_bsz 256 --num_images 50000 --cfg 1.0 \
    --output_dir ${OUTPUT_DIR} \
    --resume ${CKPT_DIR} \
    --report_to tensorboard \
    --evaluate_gen
