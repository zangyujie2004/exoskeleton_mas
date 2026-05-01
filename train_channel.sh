python3 train_channel_shap.py --greedy_forward_train --greedy_beam_width 3 \
  --greedy_out_dir checkpoints/greedy_beam3 \
  --greedy_out_json greedy_beam3.json \
  --model_type resnet18 --epochs 32 --greedy_epochs 32