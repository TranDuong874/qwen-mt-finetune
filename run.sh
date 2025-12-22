#!/bin/bash

# Train with multi-GPU
echo "Starting training with 2 GPUs..."
accelerate launch --num_processes=2 train_cpt.py --config config.yaml

# Evaluate the final model
echo "Evaluating final model..."
python evaluate.py \
  --config config.yaml \
  --adapter_model_path outputs/final_model \
  --output_dir outputs
