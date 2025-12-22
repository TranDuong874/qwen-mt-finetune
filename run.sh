accelerate launch --num_processes=2 train_cpt.py --config config.yaml

python evaluate.py --config config.yaml \
    --adapter_model_path outputs/final_model \
    --output_dir eval_results