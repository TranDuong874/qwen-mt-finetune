# Qwen Medical Translation CPT

Continuous Pre-Training (CPT) pipeline for Qwen models on Vietnamese-English medical translation with QLoRA.

## Features

- **Streaming Data**: Loads data directly from HuggingFace Hub (no local storage needed)
- **Bidirectional Translation**: EN→VI (`[VI]` prefix) and VI→EN (`[EN]` prefix)
- **QLoRA**: 4-bit quantization with LoRA adapters
- **Proper Masking**: Loss computed only on target tokens
- **Auto HF Push**: Pushes best model to HuggingFace Hub on eval improvement
- **Metrics**: Perplexity, BLEU, chrF++, COMET

## Project Structure

```
qwen-mt-finetune/
├── train_cpt.py         # Main training script
├── evaluate.py          # Evaluation with all metrics
├── config.yaml          # Production config
├── config_sanity.yaml   # Quick sanity check config
├── requirements.txt     # Dependencies
└── .env                 # API keys (create this)
```

## Quick Start (Vast.ai)

### 1. Create Instance

On [Vast.ai](https://vast.ai/), select:
- **GPU**: RTX 4090 / A100 / H100 (24GB+ VRAM recommended)
- **Image**: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel` or similar
- **Disk**: 50GB+ (for model weights)

### 2. SSH & Clone

```bash
ssh -p PORT root@HOST

# Clone repo
git clone https://github.com/YOUR_USERNAME/qwen-mt-finetune.git
cd qwen-mt-finetune
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

Create `.env` file with your API keys:

```bash
cat > .env << 'EOF'
HUGGING_FACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
EOF
```

Get tokens from:
- HuggingFace: https://huggingface.co/settings/tokens
- W&B: https://wandb.ai/authorize

### 5. Run Training

```bash
# Sanity check first (5 samples, ~1 min)
python train_cpt.py --config config_sanity.yaml

# Full training
python train_cpt.py --config config.yaml
```

### 6. Run Evaluation

```bash
python evaluate.py \
    --adapter_model_path outputs/best_model \
    --output_dir outputs \
    --test_split test
```

## Configuration

### config.yaml (Production)

```yaml
experiment_name: "qwen-cpt-medical-mt"
base_model: "Qwen/Qwen2.5-3B"

dataset:
  hf_repo: "TranDuong/medical-vlsp-2025"

training:
  max_steps: 20000
  per_device_batch_size: 16
  gradient_accumulation_steps: 2
  learning_rate: 2.0e-4
  eval_steps: 500
  save_steps: 500

huggingface:
  push_best_model: true
  repo_id: "TranDuong/qwen-medical-mt-cpt"
  private: true
```

### Key Settings

| Setting | Description | Recommended |
|---------|-------------|-------------|
| `base_model` | Qwen model to finetune | `Qwen/Qwen2.5-3B` |
| `max_steps` | Total training steps | 20000-50000 |
| `per_device_batch_size` | Batch size per GPU | 8-16 (24GB VRAM) |
| `gradient_accumulation_steps` | Effective batch multiplier | 2-4 |
| `eval_steps` | Evaluate every N steps | 500-1000 |
| `max_samples` | Limit samples (for testing) | `null` for full |

## Data Format

The dataset uses CSV with `src` and `tgt` columns:

```csv
src,tgt
"[VI] English text here","Vietnamese translation"
"[EN] Vietnamese text here","English translation"
```

- `[VI]` prefix = translate to Vietnamese
- `[EN]` prefix = translate to English

## Monitoring

### W&B Dashboard

Training logs to Weights & Biases:
- Loss curves
- Eval loss / perplexity
- Learning rate schedule

### HuggingFace Hub

Best model is automatically pushed to:
`https://huggingface.co/TranDuong/qwen-medical-mt-cpt`

## Output Structure

```
outputs/
├── checkpoint-500/      # Intermediate checkpoints
├── checkpoint-1000/
├── best_model/          # Best model (lowest eval_loss)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files
└── final_model/         # Final model after training
```

## Troubleshooting

### CUDA OOM

Reduce batch size or enable more aggressive gradient checkpointing:

```yaml
training:
  per_device_batch_size: 4
  gradient_accumulation_steps: 4
  gradient_checkpointing: true
```

### Slow Data Loading

The first epoch may be slow due to HuggingFace dataset caching. Subsequent epochs will be faster.

### Authentication Error

Make sure `.env` has valid tokens:

```bash
# Test HuggingFace auth
python -c "from huggingface_hub import HfApi; HfApi().whoami()"

# Test dataset access
python -c "from datasets import load_dataset; import os; from dotenv import load_dotenv; load_dotenv(); ds = load_dataset('TranDuong/medical-vlsp-2025', data_files={'train': 'cleaned_data/train.csv'}, split='train', streaming=True, token=os.getenv('HUGGING_FACE_TOKEN')); print(next(iter(ds)))"
```

## Metrics

| Metric | Description |
|--------|-------------|
| **Perplexity** | Lower = better language modeling |
| **BLEU** | N-gram precision (0-100) |
| **chrF++** | Character n-gram F-score with word order |
| **COMET** | Neural metric (-1 to 1, higher = better) |

## License

MIT
