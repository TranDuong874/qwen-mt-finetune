"""
Continuous Pre-Training (CPT) script for Qwen translation model.
Multi-GPU support with Accelerate, proper masking, and HuggingFace Hub integration.

Launch with: accelerate launch --num_processes=2 train_cpt.py --config config.yaml
"""
import argparse
import math
import os
from typing import Dict

import torch
import wandb
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_scheduler,
)

load_dotenv()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class TranslationDataset(Dataset):
    """
    Map-style dataset for translation with proper masking.
    Required for multi-GPU to properly shard data across processes.
    """

    def __init__(
        self,
        tokenizer,
        max_length: int = 512,
        seed: int = 42,
        hf_repo: str = None,
        local_path: str = None,
        split: str = "train",
        max_samples: int = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load dataset
        if local_path:
            self.dataset = load_dataset(
                "csv",
                data_files={split: local_path},
                split=split,
            )
        else:
            split_to_file = {
                "train": "cleaned_data/train.csv",
                "validation": "cleaned_data/val.csv",
                "test": "cleaned_data/test.csv",
            }
            self.dataset = load_dataset(
                hf_repo,
                data_files={split: split_to_file.get(split, f"cleaned_data/{split}.csv")},
                split=split,
                token=os.getenv("HUGGING_FACE_TOKEN"),
            )

        if split == "train":
            self.dataset = self.dataset.shuffle(seed=seed)

        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict:
        example = self.dataset[idx]
        return self._tokenize(example)

    def _tokenize(self, example: Dict) -> Dict:
        """Tokenize and create masked labels."""
        src = str(example.get("src", "")).strip()
        tgt = str(example.get("tgt", "")).strip()

        if not src or not tgt:
            # Return dummy data for invalid examples
            return {
                "input_ids": [self.tokenizer.pad_token_id],
                "attention_mask": [0],
                "labels": [-100],
            }

        source_ids = self.tokenizer(
            src + " ",
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        target_ids = self.tokenizer(
            tgt + self.tokenizer.eos_token,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        input_ids = source_ids + target_ids
        if len(input_ids) > self.max_length:
            max_source_len = max(self.max_length - len(target_ids), 64)
            source_ids = source_ids[:max_source_len]
            remaining = self.max_length - len(source_ids)
            target_ids = target_ids[:remaining]
            input_ids = source_ids + target_ids

        source_len = len(source_ids)
        attention_mask = [1] * len(input_ids)
        labels = [-100] * source_len + target_ids

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class DataCollator:
    """Collate batch with padding."""

    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features: list) -> Dict[str, torch.Tensor]:
        max_len = min(
            max(len(f["input_ids"]) for f in features),
            self.max_length,
        )

        input_ids = []
        attention_mask = []
        labels = []

        for f in features:
            ids = f["input_ids"][:max_len]
            mask = f["attention_mask"][:max_len]
            lbl = f["labels"][:max_len]

            pad_len = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad_len)
            attention_mask.append(mask + [0] * pad_len)
            labels.append(lbl + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def setup_wandb(config: dict, accelerator: Accelerator) -> bool:
    """Initialize W&B on main process only."""
    wandb_cfg = config.get("wandb", {})
    use_wandb = wandb_cfg.get("enabled", False) and accelerator.is_main_process

    if use_wandb:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(
            project=wandb_cfg.get("project", "qwen-mt-cpt"),
            name=config.get("experiment_name", "cpt-run"),
            config=config,
            job_type="cpt",
            tags=["continuous", "cpt", "multi-gpu"],
        )

    return use_wandb


def push_to_hub(model, tokenizer, config: dict, accelerator: Accelerator, message: str):
    """Push model to HuggingFace Hub from main process."""
    if not accelerator.is_main_process:
        return

    hf_config = config.get("huggingface", {})
    if not hf_config.get("push_best_model"):
        return

    output_dir = config.get("output_dir", "outputs")
    save_path = f"{output_dir}/best_model"

    # Unwrap and save
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # Push
    api = HfApi()
    try:
        api.create_repo(
            repo_id=hf_config["repo_id"],
            repo_type="model",
            private=hf_config.get("private", True),
            exist_ok=True,
            token=os.getenv("HUGGING_FACE_TOKEN"),
        )
    except Exception as e:
        print(f"Repo creation note: {e}")

    api.upload_folder(
        folder_path=save_path,
        repo_id=hf_config["repo_id"],
        repo_type="model",
        commit_message=message,
        token=os.getenv("HUGGING_FACE_TOKEN"),
    )
    print(f"Pushed to https://huggingface.co/{hf_config['repo_id']}")


def train(config: dict):
    """Main training function with multi-GPU support."""
    set_seed(42)

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config.get("training", {}).get("gradient_accumulation_steps", 1),
        mixed_precision="bf16" if config.get("training", {}).get("bf16", True) else "no",
    )

    use_wandb = setup_wandb(config, accelerator)

    # Quantization config (works with accelerate)
    quant_cfg = config.get("quantization", {})
    use_quantization = quant_cfg.get("load_in_4bit", False)

    if use_quantization:
        compute_dtype = getattr(torch, quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16"))
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
        )
    else:
        bnb_config = None

    # Load base model
    accelerator.print(f"Loading base model: {config['base_model']}")

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }

    if use_quantization:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = {"": accelerator.local_process_index}
    else:
        model_kwargs["device_map"] = None  # Let accelerate handle device placement

    base_model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        **model_kwargs,
    )

    # Prepare for kbit training if quantized
    if use_quantization:
        base_model = prepare_model_for_kbit_training(base_model)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config["base_model"],
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Setup LoRA
    lora_cfg = config.get("lora", {})
    adapter_path = config.get("adapter_path")

    if adapter_path and os.path.exists(adapter_path):
        accelerator.print(f"Loading existing adapter from {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True
    else:
        accelerator.print("Creating new LoRA adapter")
        lora_config = LoraConfig(
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("alpha", 32),
            target_modules=lora_cfg.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]),
            lora_dropout=lora_cfg.get("dropout", 0.0),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)

    model.enable_input_require_grads()
    if accelerator.is_main_process:
        model.print_trainable_parameters()
    model.config.pad_token_id = tokenizer.pad_token_id

    # Enable gradient checkpointing
    if config.get("training", {}).get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    # Dataset config
    dataset_cfg = config.get("dataset", {})
    hf_repo = dataset_cfg.get("hf_repo")
    local_dir = dataset_cfg.get("local_dir")
    max_samples = dataset_cfg.get("max_samples")
    train_cfg = config.get("training", {})
    max_length = train_cfg.get("max_seq_length", 512)

    # Create datasets
    if local_dir:
        accelerator.print(f"Loading dataset from local: {local_dir}")
        train_dataset = TranslationDataset(
            tokenizer=tokenizer,
            max_length=max_length,
            local_path=f"{local_dir}/train.csv",
            split="train",
            max_samples=max_samples,
        )
        eval_dataset = TranslationDataset(
            tokenizer=tokenizer,
            max_length=max_length,
            local_path=f"{local_dir}/val.csv",
            split="train",
            max_samples=max_samples,
        )
    else:
        accelerator.print(f"Loading dataset from HuggingFace: {hf_repo}")
        train_dataset = TranslationDataset(
            tokenizer=tokenizer,
            max_length=max_length,
            hf_repo=hf_repo,
            split="train",
            max_samples=max_samples,
        )
        eval_dataset = TranslationDataset(
            tokenizer=tokenizer,
            max_length=max_length,
            hf_repo=hf_repo,
            split="validation",
            max_samples=max_samples,
        )

    # Data collator
    data_collator = DataCollator(tokenizer, max_length=max_length)

    # Dataloaders with multiple workers for speed
    batch_size = train_cfg.get("per_device_batch_size", 8)
    num_workers = train_cfg.get("dataloader_num_workers", 4)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("learning_rate", 2e-4),
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    # Calculate training steps
    num_epochs = train_cfg.get("num_epochs", 3)
    num_update_steps_per_epoch = len(train_dataloader) // accelerator.gradient_accumulation_steps
    max_train_steps = num_epochs * num_update_steps_per_epoch

    # Override with max_steps if specified (for fractional epochs)
    if train_cfg.get("max_steps"):
        max_train_steps = train_cfg["max_steps"]
        num_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    warmup_steps = int(max_train_steps * train_cfg.get("warmup_ratio", 0.03))

    # Scheduler
    lr_scheduler = get_scheduler(
        name=train_cfg.get("scheduler", "cosine"),
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )

    # Prepare for distributed training
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    accelerator.print(f"\n{'='*60}")
    accelerator.print(f"Training Configuration:")
    accelerator.print(f"  Num GPUs: {accelerator.num_processes}")
    accelerator.print(f"  Batch size per GPU: {batch_size}")
    accelerator.print(f"  Gradient accumulation steps: {accelerator.gradient_accumulation_steps}")
    accelerator.print(f"  Effective batch size: {batch_size * accelerator.num_processes * accelerator.gradient_accumulation_steps}")
    accelerator.print(f"  Num epochs: {num_epochs}")
    accelerator.print(f"  Total optimization steps: {max_train_steps}")
    accelerator.print(f"  Warmup steps: {warmup_steps}")
    accelerator.print(f"{'='*60}\n")

    # Training loop
    output_dir = config.get("output_dir", "outputs")
    logging_steps = train_cfg.get("logging_steps", 50)
    eval_steps = train_cfg.get("eval_steps", 500)
    save_steps = train_cfg.get("save_steps", 500)

    best_eval_loss = float("inf")
    global_step = 0
    total_loss = 0

    accelerator.print("Starting training...")

    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            disable=not accelerator.is_main_process,
        )

        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Only count global steps after gradient accumulation
            if accelerator.sync_gradients:
                global_step += 1
                total_loss += loss.detach().float()

                # Early stop if max_steps reached
                if global_step >= max_train_steps:
                    break

                # Logging
                if global_step % logging_steps == 0:
                    avg_loss = total_loss / logging_steps
                    lr = lr_scheduler.get_last_lr()[0]

                    if accelerator.is_main_process:
                        progress_bar.set_postfix({
                            "loss": f"{avg_loss:.4f}",
                            "lr": f"{lr:.2e}",
                            "step": global_step,
                        })

                        if use_wandb:
                            wandb.log({
                                "train/loss": avg_loss,
                                "train/learning_rate": lr,
                                "train/epoch": epoch + (step / len(train_dataloader)),
                                "train/global_step": global_step,
                            })

                    total_loss = 0

                # Evaluation
                if global_step % eval_steps == 0:
                    model.eval()
                    eval_loss = 0
                    eval_steps_count = 0

                    with torch.no_grad():
                        for eval_batch in tqdm(
                            eval_dataloader,
                            desc="Evaluating",
                            disable=not accelerator.is_main_process,
                        ):
                            outputs = model(**eval_batch)
                            eval_loss += outputs.loss.detach().float()
                            eval_steps_count += 1

                    eval_loss = eval_loss / eval_steps_count
                    eval_loss = accelerator.gather(eval_loss).mean().item()
                    perplexity = math.exp(eval_loss)

                    accelerator.print(f"\nStep {global_step}: eval_loss={eval_loss:.4f}, perplexity={perplexity:.2f}")

                    if accelerator.is_main_process and use_wandb:
                        wandb.log({
                            "eval/loss": eval_loss,
                            "eval/perplexity": perplexity,
                            "eval/global_step": global_step,
                        })

                    # Save best model
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        accelerator.print(f"New best eval_loss: {eval_loss:.4f}")

                        accelerator.wait_for_everyone()
                        push_to_hub(
                            model, tokenizer, config, accelerator,
                            f"Best model - eval_loss: {eval_loss:.4f}, perplexity: {perplexity:.2f}"
                        )

                    model.train()

                # Save checkpoint
                if global_step % save_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        save_path = f"{output_dir}/checkpoint-{global_step}"
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(save_path)
                        tokenizer.save_pretrained(save_path)
                        accelerator.print(f"Saved checkpoint to {save_path}")

        # Break epoch loop if max_steps reached
        if global_step >= max_train_steps:
            break

    # Final evaluation
    accelerator.print("\nRunning final evaluation...")
    model.eval()
    eval_loss = 0
    eval_steps_count = 0

    with torch.no_grad():
        for eval_batch in tqdm(
            eval_dataloader,
            desc="Final Evaluation",
            disable=not accelerator.is_main_process,
        ):
            outputs = model(**eval_batch)
            eval_loss += outputs.loss.detach().float()
            eval_steps_count += 1

    eval_loss = eval_loss / eval_steps_count
    eval_loss = accelerator.gather(eval_loss).mean().item()
    perplexity = math.exp(eval_loss)

    accelerator.print(f"\nFinal eval_loss: {eval_loss:.4f}, perplexity: {perplexity:.2f}")

    if accelerator.is_main_process and use_wandb:
        wandb.log({
            "final/eval_loss": eval_loss,
            "final/perplexity": perplexity,
        })

    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_path = f"{output_dir}/final_model"
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        accelerator.print(f"Final model saved to {final_path}")

    # Push final model
    push_to_hub(
        model, tokenizer, config, accelerator,
        f"Final model - eval_loss: {eval_loss:.4f}, perplexity: {perplexity:.2f}"
    )

    if use_wandb:
        wandb.finish()

    accelerator.print("\nTraining complete!")


def main():
    parser = argparse.ArgumentParser(description="Qwen CPT Training (Multi-GPU)")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config YAML file",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Create output directory
    os.makedirs(config.get("output_dir", "outputs"), exist_ok=True)

    train(config)


if __name__ == "__main__":
    main()
