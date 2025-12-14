# Qwen Medical Translation Finetuning

QLoRA finetuning pipeline for Qwen models on Vietnamese-English medical translation with curriculum learning.

## Features

- **Curriculum Learning**: 10-stage progressive training (100K samples per stage)
- **QLoRA**: 4-bit quantization with LoRA adapters (~70MB per checkpoint)
- **Multi-GPU**: Distributed training with Accelerate
- **Early Stopping**: Stops training when BLEU stops improving
- **W&B Logging**: Separate training runs + unified eval run with line charts
- **HuggingFace Push**: Auto-upload best model to HF Hub

## Project Structure

```
qwen-mt-finetune/
├── config.yaml          # All experiment settings
├── orchestrator.py      # Main pipeline controller
├── train.py             # Training script (per stage)
├── evaluate.py          # Evaluation with BLEU/chrF++/COMET
├── .env                 # API keys (HF_TOKEN, WANDB_API_KEY)
└── outputs/             # Checkpoints and results
```

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch transformers peft accelerate bitsandbytes
pip install datasets wandb sacrebleu comet-ml huggingface_hub
pip install python-dotenv pyyaml tqdm

# Configure API keys in .env
HUGGING_FACE_TOKEN=hf_xxx
WANDB_API_KEY=xxx
```

## Configuration

Edit `config.yaml` to customize:

```yaml
# Key settings
experiment_name: "qwen-medical-mt"
base_model: "Qwen/Qwen3-1.7B"

# Training
training:
  num_gpus: 2
  per_device_batch_size: 1
  learning_rate: 1.0e-4
  num_epochs: 1

# Dataset
dataset:
  num_parts: 10
  max_train_samples: null  # null = full, int = limit

# Early stopping
early_stopping:
  patience: 2
  min_delta: 0.5

# HuggingFace
huggingface:
  push_best_model: true
  repo_id: "YourUsername/model-name"
  private: true
```

## Usage

### Full Training Run

```bash
source .venv/bin/activate
python orchestrator.py --config config.yaml
```

### Smoke Test (quick validation)

Edit `config.yaml`:
```yaml
dataset:
  num_parts: 2
  max_train_samples: 10
  max_test_samples: 10
training:
  save_steps: 5
  eval_steps: 5
```

Then run:
```bash
python orchestrator.py --config config.yaml
```

## W&B Dashboard

Training creates grouped runs under your experiment name:

| Run | Type | Metrics |
|-----|------|---------|
| `{name}-part1` | train | Loss graph for stage 1 |
| `{name}-part2` | train | Loss graph for stage 2 |
| ... | ... | ... |
| `{name}-eval` | eval | BLEU/COMET/chrF++ line charts across all stages |

All runs are grouped together for easy comparison.

## Output Structure

```
outputs/
├── {name}-part1/
│   └── best_model/          # LoRA adapter + tokenizer
├── {name}-part2/
│   └── best_model/
├── part-1-examples.json     # 100 random translation samples
├── part-2-examples.json
└── progress.json            # Training history & best checkpoint
```

## Metrics

- **BLEU**: SacreBLEU corpus-level score
- **chrF++**: Character n-gram F-score with word order
- **COMET**: Neural reference-based metric (Unbabel/wmt22-comet-da)

## Pipeline Flow

```
Stage 1: Base Model → Train → Eval → Save Adapter
                                ↓
Stage 2: Load Stage 1 Adapter → Train → Eval → Save Adapter
                                ↓
Stage 3: Load Stage 2 Adapter → Train → Eval → Save Adapter
                                ↓
         ... (continues until early stopping or 10 stages)
                                ↓
         Push Best Model to HuggingFace
```

## Dataset Format

```
[EN] English text here [VI] Vietnamese translation here
[EN] Another sentence [VI] Another translation
```

## License

MIT
