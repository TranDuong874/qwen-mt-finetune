# train.py
from datasets import load_dataset
import os
from dotenv import load_dotenv
import wandb
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)

from peft import LoraConfig, get_peft_model, PeftModel
import torch.distributed as dist
import argparse

load_dotenv()


def train(
    base_model_path,
    train_data_path,
    valid_data_path,
    output_dir,
    run_name,
    wandb_project,
    adapter_model_path=None,
):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # W&B init on rank 0 only
    if local_rank == 0:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(project=wandb_project, name=run_name)

    # Load dataset
    dataset = load_dataset(
        "text",
        data_files={"train": train_data_path, "valid": valid_data_path},
    )
    # Subsample for smoke test
    dataset["train"] = dataset["train"].select(range(min(10, len(dataset["train"]))))
    dataset["valid"] = dataset["valid"].select(range(min(10, len(dataset["valid"]))))

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
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
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # Tokenizer setup
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    # Tokenize dataset
    def tokenize(batch):
        texts = [t.strip() + " " + tokenizer.eos_token for t in batch["text"]]
        return tokenizer(texts, truncation=True, max_length=512)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        learning_rate=1e-4,
        warmup_ratio=0.0,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=2,
        save_steps=2,
        fp16=True,
        report_to="wandb" if local_rank == 0 else "none",
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        load_best_model_at_end=False,
        save_total_limit=1,
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
        trainer.save_model(f"{output_dir}/best_model")
        tokenizer.save_pretrained(f"{output_dir}/best_model")
        wandb.finish()

    # Cleanup DDP
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--valid_data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--wandb_project", type=str, required=True)
    parser.add_argument("--adapter_model_path", type=str, default=None)
    args = parser.parse_args()

    train(
        args.base_model_path,
        args.train_data_path,
        args.valid_data_path,
        args.output_dir,
        args.run_name,
        args.wandb_project,
        args.adapter_model_path,
    )