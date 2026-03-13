mkdir -p /root/.cache/torch/hub/checkpoints/
cp /lpai/volumes/so-volume-bd-ga/lhp/pt_inception-2015-12-05-6726825d.pth /root/.cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth

CUDA_VISIBLE_DEVICES=6,7 \
torchrun --nproc_per_node=2 main_jit.py --config configs/irepa.yaml \
  --model JiT-B/16 \
  --enc_type="dinov3-vit-b16" \
  --encoder_depth=4 \
  --data_path=../data \
  --epochs 1000 \
  --max_train_steps 100 \
  --output_dir="/lpai/output/models/jit-dinov3-vit-b16-irepa" \
  --report_to tensorboard \

