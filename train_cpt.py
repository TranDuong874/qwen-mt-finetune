"""
Continuous Pre-Training (CPT) script for Qwen translation model.
Supports streaming data, proper masking, evaluation, and HuggingFace Hub integration.
"""
import argparse
import math
import os
from typing import Dict

import torch
import wandb
import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

load_dotenv()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class TranslationDataset(Dataset):
    """
    Map-style dataset for translation with proper masking.
    Masks source text + language token, computes loss only on target.
    Supports both HuggingFace repos and local CSV files.
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

        # Load dataset - either from HF or local CSV
        if local_path:
            self.dataset = load_dataset(
                "csv",
                data_files={split: local_path},
                split=split,
            )
        else:
            # Map split names to CSV file paths
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

        # Tokenize source (includes [EN]/[VI] token) and target separately
        source_ids = self.tokenizer(
            src + " ",  # Space separator
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        target_ids = self.tokenizer(
            tgt + self.tokenizer.eos_token,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        # Concatenate and truncate if needed
        input_ids = source_ids + target_ids
        if len(input_ids) > self.max_length:
            # Prioritize keeping target, truncate source if needed
            max_source_len = max(self.max_length - len(target_ids), 64)
            source_ids = source_ids[:max_source_len]
            remaining = self.max_length - len(source_ids)
            target_ids = target_ids[:remaining]
            input_ids = source_ids + target_ids

        source_len = len(source_ids)
        attention_mask = [1] * len(input_ids)

        # Labels: -100 for source tokens (masked), actual ids for target
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
        # Find max length in batch
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

            # Pad to max_len
            pad_len = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad_len)
            attention_mask.append(mask + [0] * pad_len)
            labels.append(lbl + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class HuggingFaceCallback(TrainerCallback):
    """Callback to push best model to HuggingFace Hub."""

    def __init__(self, config: dict, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.best_loss = float("inf")
        self.hf_config = config.get("huggingface", {})
        self.api = HfApi() if self.hf_config.get("push_best_model") else None

        # Create repo if needed
        if self.api and self.hf_config.get("push_best_model"):
            try:
                self.api.create_repo(
                    repo_id=self.hf_config["repo_id"],
                    repo_type="model",
                    private=self.hf_config.get("private", True),
                    exist_ok=True,
                    token=os.getenv("HUGGING_FACE_TOKEN"),
                )
            except Exception as e:
                print(f"Repo creation note: {e}")

    def on_evaluate(self, args, state, control, metrics, **kwargs):  # noqa: ARG002
        """Push model if eval loss improved."""
        if not self.api or not self.hf_config.get("push_best_model"):
            return

        eval_loss = metrics.get("eval_loss", float("inf"))
        if eval_loss < self.best_loss:
            self.best_loss = eval_loss
            print(f"\nNew best eval_loss: {eval_loss:.4f}, pushing to HuggingFace...")

            # Save model temporarily
            model = kwargs.get("model")
            if model:
                save_path = f"{args.output_dir}/best_model"
                model.save_pretrained(save_path)
                self.tokenizer.save_pretrained(save_path)

                # Push to hub
                self.api.upload_folder(
                    folder_path=save_path,
                    repo_id=self.hf_config["repo_id"],
                    repo_type="model",
                    commit_message=f"Best model - eval_loss: {eval_loss:.4f}",
                    token=os.getenv("HUGGING_FACE_TOKEN"),
                )
                print(f"Pushed to https://huggingface.co/{self.hf_config['repo_id']}")


def setup_wandb(config: dict) -> bool:
    """Initialize W&B if enabled."""
    wandb_cfg = config.get("wandb", {})
    use_wandb = wandb_cfg.get("enabled", False)

    if use_wandb:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(
            project=wandb_cfg.get("project", "qwen-mt-cpt"),
            name=config.get("experiment_name", "cpt-run"),
            config=config,
            job_type="cpt",
            tags=["continuous", "cpt"],
        )

    return use_wandb


def train(config: dict):
    """Main training function."""
    use_wandb = setup_wandb(config)

    # Quantization config
    quant_cfg = config.get("quantization", {})
    compute_dtype = getattr(torch, quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16"))
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
    )

    # Load base model
    print(f"Loading base model: {config['base_model']}")
    # When using accelerate with quantization, use current_device
    base_model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        quantization_config=bnb_config,
        device_map={'': torch.cuda.current_device()},
        trust_remote_code=True,
    )

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
        print(f"Loading existing adapter from {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True
    else:
        print("Creating new LoRA adapter")
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
    model.print_trainable_parameters()
    model.config.pad_token_id = tokenizer.pad_token_id

    # Dataset config
    dataset_cfg = config.get("dataset", {})
    hf_repo = dataset_cfg.get("hf_repo")
    local_dir = dataset_cfg.get("local_dir")
    max_samples = dataset_cfg.get("max_samples")  # For sanity checking
    train_cfg = config.get("training", {})
    max_length = train_cfg.get("max_seq_length", 512)

    # Create streaming datasets
    if local_dir:
        print(f"Loading dataset from local: {local_dir}")
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
            split="train",  # CSV loader uses 'train' split
            max_samples=max_samples,
        )
    else:
        print(f"Loading dataset from HuggingFace: {hf_repo}")
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

    # Training arguments
    output_dir = config.get("output_dir", "outputs")
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=config.get("experiment_name", "cpt-run"),
        # Batch settings
        per_device_train_batch_size=train_cfg.get("per_device_batch_size", 8),
        per_device_eval_batch_size=train_cfg.get("per_device_batch_size", 8),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        # Training duration
        num_train_epochs=train_cfg.get("num_epochs", 3),
        # Learning rate
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        lr_scheduler_type=train_cfg.get("scheduler", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        # Precision
        fp16=train_cfg.get("fp16", False),
        bf16=train_cfg.get("bf16", True),
        # Logging & Eval
        logging_steps=train_cfg.get("logging_steps", 50),
        eval_strategy="steps",
        eval_steps=train_cfg.get("eval_steps", 500),
        # Saving
        save_strategy="steps",
        save_steps=train_cfg.get("save_steps", 500),
        save_total_limit=train_cfg.get("save_total_limit", 2),
        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Other
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        report_to="wandb" if use_wandb else "none",
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 4),
    )

    # Callbacks
    callbacks = [HuggingFaceCallback(config, tokenizer)]

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Final evaluation
    print("\nRunning final evaluation...")
    final_metrics = trainer.evaluate()

    # Compute perplexity from loss
    if "eval_loss" in final_metrics:
        perplexity = math.exp(final_metrics["eval_loss"])
        final_metrics["eval_perplexity"] = perplexity
        print(f"Final perplexity: {perplexity:.2f}")

    if use_wandb:
        wandb.log({"final/perplexity": final_metrics.get("eval_perplexity", 0)})
        wandb.log({"final/eval_loss": final_metrics.get("eval_loss", 0)})

    # Save final model
    final_path = f"{output_dir}/final_model"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nFinal model saved to {final_path}")

    # Final push to HuggingFace
    hf_config = config.get("huggingface", {})
    if hf_config.get("push_best_model"):
        print("\nPushing final model to HuggingFace Hub...")
        api = HfApi()
        api.upload_folder(
            folder_path=final_path,
            repo_id=hf_config["repo_id"],
            repo_type="model",
            commit_message=f"Final model - perplexity: {final_metrics.get('eval_perplexity', 0):.2f}",
            token=os.getenv("HUGGING_FACE_TOKEN"),
        )
        print(f"Model pushed to https://huggingface.co/{hf_config['repo_id']}")

    if use_wandb:
        wandb.finish()

    print("\nTraining complete!")
    return final_metrics


def main():
    parser = argparse.ArgumentParser(description="Qwen CPT Training")
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
