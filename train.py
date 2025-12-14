# train.py
"""
Training script for Qwen finetuning with QLoRA.
Supports curriculum learning with adapter chaining.
"""
import argparse
import os

import torch
import wandb
import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
import torch.distributed as dist

load_dotenv()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train(
    config: dict,
    train_data_path: str,
    valid_data_path: str,
    output_dir: str,
    run_name: str,
    part: int,
    adapter_model_path: str = None,
):
    """Train model on a single curriculum part."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # W&B: Each training stage gets its own run with loss graphs
    use_wandb = config["wandb"]["enabled"] and local_rank == 0

    if use_wandb:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(
            project=config["wandb"]["project"],
            name=run_name,
            config=config,
            group=config["experiment_name"],  # Group all stages together
            job_type="train",
            tags=[f"part-{part}", "training"],
        )

    # Load dataset
    dataset = load_dataset(
        "text",
        data_files={"train": train_data_path, "valid": valid_data_path},
    )

    # Apply sample limits if configured
    max_train = config["dataset"]["max_train_samples"]
    max_eval = config["dataset"]["max_eval_samples"]
    if max_train:
        dataset["train"] = dataset["train"].select(
            range(min(max_train, len(dataset["train"])))
        )
    if max_eval:
        dataset["valid"] = dataset["valid"].select(
            range(min(max_eval, len(dataset["valid"])))
        )

    print(f"Train samples: {len(dataset['train'])}, Valid samples: {len(dataset['valid'])}")

    # Quantization config
    quant_cfg = config["quantization"]
    compute_dtype = getattr(torch, quant_cfg["bnb_4bit_compute_dtype"])
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
    )

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        quantization_config=bnb_config,
        device_map={"": local_rank},
    )

    # Load adapter or create new LoRA
    if adapter_model_path:
        print(f"Loading adapter from {adapter_model_path}")
        model = PeftModel.from_pretrained(base_model, adapter_model_path)
        tokenizer = AutoTokenizer.from_pretrained(adapter_model_path)
        # Ensure LoRA layers are trainable
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True
    else:
        print("Creating new LoRA adapter")
        lora_cfg = config["lora"]
        lora_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            target_modules=lora_cfg["target_modules"],
            lora_dropout=lora_cfg["dropout"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)
        tokenizer = AutoTokenizer.from_pretrained(config["base_model"])

    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # Tokenizer setup
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    # Tokenize dataset
    max_length = config["training"]["max_seq_length"]

    def tokenize(batch):
        texts = [t.strip() + " " + tokenizer.eos_token for t in batch["text"]]
        return tokenizer(texts, truncation=True, max_length=max_length)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    # Training args from config
    train_cfg = config["training"]
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        per_device_train_batch_size=train_cfg["per_device_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        num_train_epochs=train_cfg["num_epochs"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        logging_steps=train_cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=train_cfg["eval_steps"],
        save_steps=train_cfg["save_steps"],
        fp16=train_cfg["fp16"],
        report_to="wandb" if use_wandb else "none",
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        ddp_find_unused_parameters=False,
        load_best_model_at_end=False,
        save_total_limit=train_cfg["save_total_limit"],
        save_only_model=True,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["valid"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # Train
    trainer.train()

    # Save on rank 0 only
    if local_rank == 0:
        save_path = f"{output_dir}/best_model"
        trainer.save_model(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")

        # Finish W&B run
        if use_wandb:
            wandb.finish()

    # Cleanup DDP
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Qwen Finetuning Trainer")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML file")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--valid_data_path", type=str, required=True, help="Path to validation data")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--run_name", type=str, required=True, help="Run name for logging")
    parser.add_argument("--part", type=int, required=True, help="Curriculum part number")
    parser.add_argument("--adapter_model_path", type=str, default=None, help="Path to existing adapter")
    args = parser.parse_args()

    config = load_config(args.config)

    train(
        config=config,
        train_data_path=args.train_data_path,
        valid_data_path=args.valid_data_path,
        output_dir=args.output_dir,
        run_name=args.run_name,
        part=args.part,
        adapter_model_path=args.adapter_model_path,
    )


if __name__ == "__main__":
    main()
