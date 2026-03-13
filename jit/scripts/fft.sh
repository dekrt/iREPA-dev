CUDA_VISIBLE_DEVICES=5,6 \
torchrun --nproc_per_node=2 main_fft.py --config configs/irepa.yaml \
  --model JiT-B/16 \
  --enc_type="dinov3-vit-b16" \
  --encoder_depth=4 \
  --data_path=../data \
  --epochs 200 \
  --output_dir="/lpai/output/models/jit-dinov3-vit-b16-irepa-fft-l2" \
  --report_to tensorboard \
  --projection_loss_type freq_l2\
  --freq_radius 4 \

