cd /data/zyj/project_mas
python3 train_gradcam.py \
  --checkpoint_path checkpoints/resnet_spasm_best.pt \
  --pickle_path init_window_cache.pkl \
  --out_dir window_vis/gradcam_train \
  --num_samples 24 \
  --periods 0,1,7 \
  --emg_channels 0,6
  