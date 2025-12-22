"""
GRPO (Group Relative Policy Optimization) training for Qwen translation model.
Uses reward functions to improve translation quality beyond supervised learning.

Launch with: accelerate launch train_grpo.py --config config_grpo.yaml
"""
import argparse
import gc
import math
import os
import time
from typing import Dict, List, Callable

import torch
import wandb
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi
from peft import PeftModel, prepare_model_for_kbit_training
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_scheduler,
)

load_dotenv()


# =============================================================================
# REWARD FUNCTIONS
# =============================================================================

def reward_bleu(
    sources: List[str],
    generations: List[str],
    references: List[str],
) -> List[float]:
    """
    Compute sentence-level BLEU scores.
    Returns list of scores in [0, 1] range.
    """
    from sacrebleu import sentence_bleu

    scores = []
    for gen, ref in zip(generations, references):
        if not gen.strip():
            scores.append(0.0)
            continue
        try:
            bleu = sentence_bleu(gen, [ref])
            scores.append(bleu.score / 100.0)  # Normalize to [0, 1]
        except Exception:
            scores.append(0.0)
    return scores


def reward_comet(
    sources: List[str],
    generations: List[str],
    references: List[str],
    comet_model=None,
    batch_size: int = 32,
) -> List[float]:
    """
    Compute COMET scores for generated translations.
    Returns list of scores in [0, 1] range.
    """
    if comet_model is None:
        return [0.5] * len(sources)  # Fallback

    # Clean sources (remove language tags)
    clean_sources = []
    for s in sources:
        if s.startswith("[VI] ") or s.startswith("[EN] "):
            clean_sources.append(s[5:])
        else:
            clean_sources.append(s)

    comet_data = [
        {"src": src, "mt": gen, "ref": ref}
        for src, gen, ref in zip(clean_sources, generations, references)
    ]

    output = comet_model.predict(comet_data, batch_size=batch_size, gpus=1)
    return output.scores


def reward_length_ratio(
    sources: List[str],
    generations: List[str],
    references: List[str],
    min_ratio: float = 0.5,
    max_ratio: float = 1.5,
) -> List[float]:
    """
    Penalize generations that are too short or too long compared to reference.
    Returns 1.0 for good length, decreasing penalty otherwise.
    """
    scores = []
    for gen, ref in zip(generations, references):
        gen_len = len(gen.split())
        ref_len = len(ref.split())

        if ref_len == 0:
            scores.append(0.5)
            continue

        ratio = gen_len / ref_len

        if min_ratio <= ratio <= max_ratio:
            # Good range - small penalty based on distance from 1.0
            score = 1.0 - 0.2 * abs(ratio - 1.0)
        elif ratio < min_ratio:
            # Too short
            score = max(0.0, ratio / min_ratio * 0.5)
        else:
            # Too long
            score = max(0.0, 1.0 - (ratio - max_ratio) * 0.3)

        scores.append(score)

    return scores


def reward_no_copy(
    sources: List[str],
    generations: List[str],
    references: List[str],
    threshold: float = 0.8,
) -> List[float]:
    """
    Penalize generations that are too similar to source (copying instead of translating).
    """
    scores = []
    for src, gen in zip(sources, generations):
        # Clean source
        clean_src = src[5:] if src.startswith("[VI] ") or src.startswith("[EN] ") else src

        # Simple overlap check
        src_words = set(clean_src.lower().split())
        gen_words = set(gen.lower().split())

        if len(gen_words) == 0:
            scores.append(0.0)
            continue

        overlap = len(src_words & gen_words) / len(gen_words)

        if overlap > threshold:
            # High overlap = copying, penalize
            scores.append(max(0.0, 1.0 - overlap))
        else:
            scores.append(1.0)

    return scores


def reward_not_empty(
    sources: List[str],
    generations: List[str],
    references: List[str],
    min_words: int = 2,
) -> List[float]:
    """
    Heavily penalize empty or very short generations.
    """
    scores = []
    for gen in generations:
        words = gen.strip().split()
        if len(words) < min_words:
            scores.append(0.0)
        else:
            scores.append(1.0)
    return scores


# =============================================================================
# NUMERIC ACCURACY REWARD (for medical translation)
# =============================================================================

def _normalize_number(s: str) -> float:
    """Convert string to float, handling various formats (US/EU)."""
    s = s.replace(' ', '')
    if '.' in s and ',' in s:
        if s.rfind(',') > s.rfind('.'):
            s = s.replace('.', '').replace(',', '.')
        else:
            s = s.replace(',', '')
    elif ',' in s:
        parts = s.split(',')
        if len(parts) == 2 and len(parts[1]) <= 2:
            s = s.replace(',', '.')
        else:
            s = s.replace(',', '')
    try:
        return float(s)
    except:
        return float('nan')


def _get_numbers_from_text(text: str) -> set:
    """Extract normalized numbers from text."""
    import re
    patterns = [
        r'[pP]\s*[<>=≤≥]\s*(\d+[.,]\d+)',  # p-values
        r'(\d+[.,]?\d*)\s*%',               # percentages
        r'(?<![.,\d])(\d+[.,]\d+)(?![.,\d])', # decimals
        r'(?<![.,\d])(\d+)(?![.,\d%])',      # integers
    ]
    numbers = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            raw = match.group(1)
            norm = _normalize_number(raw)
            if not (norm != norm):  # not NaN
                numbers.add(norm)
    return numbers


def reward_numeric_accuracy(
    sources: List[str],
    generations: List[str],
    references: List[str],
    missing_penalty: float = 0.3,
    hallucination_penalty: float = 0.2,
    max_penalty: float = 0.5,
) -> List[float]:
    """
    Compute numeric accuracy reward for medical translation.

    Penalizes:
    - Missing numbers (from source that don't appear in generation)
    - Hallucinated numbers (in generation but not in source or reference)

    Returns scores in [0, 1] where 1.0 = perfect numeric preservation.
    """
    scores = []
    for src, gen, ref in zip(sources, generations, references):
        src_nums = _get_numbers_from_text(src)
        gen_nums = _get_numbers_from_text(gen)
        ref_nums = _get_numbers_from_text(ref)

        if len(src_nums) == 0:
            scores.append(1.0)
            continue

        missing = src_nums - gen_nums
        hallucinated = gen_nums - src_nums - ref_nums

        penalty = (
            missing_penalty * len(missing) / len(src_nums) +
            hallucination_penalty * len(hallucinated) / max(1, len(src_nums))
        )
        scores.append(max(0.0, 1.0 - min(penalty, max_penalty)))

    return scores


class RewardCombiner:
    """Combine multiple reward functions with weights."""

    def __init__(
        self,
        comet_model=None,
        use_comet: bool = True,
        use_bleu: bool = False,
        use_numeric: bool = True,
        comet_weight: float = 0.55,
        bleu_weight: float = 0.0,
        length_weight: float = 0.1,
        no_copy_weight: float = 0.1,
        not_empty_weight: float = 0.1,
        numeric_weight: float = 0.15,
    ):
        self.comet_model = comet_model
        self.use_comet = use_comet
        self.use_bleu = use_bleu
        self.use_numeric = use_numeric
        self.weights = {
            "comet": comet_weight if use_comet else 0.0,
            "bleu": bleu_weight if use_bleu else 0.0,
            "length": length_weight,
            "no_copy": no_copy_weight,
            "not_empty": not_empty_weight,
            "numeric": numeric_weight if use_numeric else 0.0,
        }

    def __call__(
        self,
        sources: List[str],
        generations: List[str],
        references: List[str],
    ) -> Dict[str, List[float]]:
        """Compute all rewards and combined score."""
        rewards = {}

        # Individual rewards
        if self.use_comet:
            rewards["comet"] = reward_comet(
                sources, generations, references, self.comet_model
            )
        else:
            rewards["comet"] = [0.0] * len(sources)

        if self.use_bleu:
            rewards["bleu"] = reward_bleu(sources, generations, references)
        else:
            rewards["bleu"] = [0.0] * len(sources)

        rewards["length"] = reward_length_ratio(sources, generations, references)
        rewards["no_copy"] = reward_no_copy(sources, generations, references)
        rewards["not_empty"] = reward_not_empty(sources, generations, references)

        # Numeric accuracy reward (important for medical translation)
        if self.use_numeric:
            rewards["numeric"] = reward_numeric_accuracy(
                sources, generations, references
            )
        else:
            rewards["numeric"] = [0.0] * len(sources)

        # Combined reward
        combined = []
        for i in range(len(sources)):
            score = sum(
                self.weights[k] * rewards[k][i]
                for k in self.weights
            )
            combined.append(score)

        rewards["combined"] = combined
        return rewards


# =============================================================================
# DATASET
# =============================================================================

class GRPODataset(Dataset):
    """Dataset for GRPO training with prompt-reference pairs."""

    def __init__(
        self,
        tokenizer,
        max_length: int = 256,
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
            self.dataset = load_dataset(
                hf_repo,
                data_files={split: "grpo_dataset.csv"},
                split=split,
                token=os.getenv("HUGGING_FACE_TOKEN"),
            )

        if max_samples:
            self.dataset = self.dataset.select(
                range(min(max_samples, len(self.dataset)))
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict:
        example = self.dataset[idx]

        src = str(example.get("src", example.get("source", ""))).strip()
        tgt = str(example.get("tgt", example.get("reference", ""))).strip()

        # Tokenize prompt (source + space to trigger generation)
        prompt = src + " "
        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]

        return {
            "prompt_ids": prompt_ids,
            "prompt": prompt,
            "source": src,
            "reference": tgt,
        }


class GRPOCollator:
    """Collate GRPO batch with left-padding for generation."""

    def __init__(self, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features: list) -> Dict:
        max_len = min(
            max(len(f["prompt_ids"]) for f in features),
            self.max_length,
        )

        # Left-pad for generation
        input_ids = []
        attention_mask = []

        for f in features:
            ids = f["prompt_ids"][-max_len:]  # Truncate from left if needed
            pad_len = max_len - len(ids)
            input_ids.append([self.pad_token_id] * pad_len + ids)
            attention_mask.append([0] * pad_len + [1] * len(ids))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "prompts": [f["prompt"] for f in features],
            "sources": [f["source"] for f in features],
            "references": [f["reference"] for f in features],
        }


# =============================================================================
# TRAINING
# =============================================================================

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_responses(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_generations: int = 4,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> List[List[str]]:
    """
    Generate multiple responses per prompt for GRPO (batched).
    Returns: list of lists, each inner list contains num_generations responses.
    """
    batch_size = input_ids.shape[0]

    # Batch all generations: [B, L] -> [B * num_gen, L]
    expanded_ids = input_ids.repeat_interleave(num_generations, dim=0)
    expanded_mask = attention_mask.repeat_interleave(num_generations, dim=0)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=expanded_ids,
            attention_mask=expanded_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode and reshape back to [B, num_gen]
    input_len = input_ids.shape[1]
    all_responses = [[] for _ in range(batch_size)]

    for i, output in enumerate(outputs):
        generated = output[input_len:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        if text.startswith("[VI] "):
            text = text[5:]
        elif text.startswith("[EN] "):
            text = text[5:]
        batch_idx = i // num_generations
        all_responses[batch_idx].append(text)

    return all_responses


def compute_grpo_loss(
    model,
    tokenizer,
    prompts: List[str],
    responses: List[List[str]],
    rewards: List[List[float]],
    device,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    Compute GRPO loss using group-relative rewards.

    For each prompt, we have multiple responses with rewards.
    We normalize rewards within each group and weight the log-probs accordingly.
    """
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    count = 0

    for prompt, resps, rews in zip(prompts, responses, rewards):
        if len(resps) == 0:
            continue

        # Normalize rewards within group (subtract mean, divide by std)
        rews_tensor = torch.tensor(rews, dtype=torch.float32, device=device)
        mean_rew = rews_tensor.mean()
        std_rew = rews_tensor.std() + 1e-8
        normalized_rews = (rews_tensor - mean_rew) / std_rew

        for resp, norm_rew in zip(resps, normalized_rews):
            # Tokenize full sequence
            full_text = prompt + resp + tokenizer.eos_token
            inputs = tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(device)

            # Forward pass WITH gradients for policy learning
            outputs = model(**inputs)
            logits = outputs.logits

            # Compute log prob of response tokens only
            prompt_ids = tokenizer(
                prompt, add_special_tokens=False
            )["input_ids"]
            prompt_len = len(prompt_ids)

            # Shift for next token prediction
            shift_logits = logits[:, prompt_len:-1, :]
            shift_labels = inputs["input_ids"][:, prompt_len + 1:]

            if shift_logits.shape[1] == 0:
                continue  # Skip if no response tokens

            # Cross entropy per token
            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(
                -1, shift_labels.unsqueeze(-1)
            ).squeeze(-1)

            # Sum log probs for sequence
            seq_log_prob = token_log_probs.sum()

            # GRPO loss: -reward * log_prob (detach norm_rew to avoid grad issues)
            loss = -norm_rew.detach() * seq_log_prob * beta
            total_loss = total_loss + loss
            count += 1

    if count == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return total_loss / count


def train(config: dict):
    """Main GRPO training function."""
    set_seed(42)

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config.get("training", {}).get(
            "gradient_accumulation_steps", 4
        ),
        mixed_precision="bf16" if config.get("training", {}).get("bf16", True) else "no",
    )

    # Setup W&B
    wandb_cfg = config.get("wandb", {})
    use_wandb = wandb_cfg.get("enabled", False) and accelerator.is_main_process
    if use_wandb:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(
            project=wandb_cfg.get("project", "qwen-mt-grpo"),
            name=config.get("experiment_name", "grpo-run"),
            config=config,
        )

    # Load COMET model for rewards (all processes need it for reward computation)
    reward_cfg = config.get("reward", {})
    comet_model = None
    if reward_cfg.get("use_comet", True):
        accelerator.print("Loading COMET model for rewards...")
        from comet import download_model, load_from_checkpoint
        comet_path = download_model(
            reward_cfg.get("comet_model", "Unbabel/wmt22-comet-da")
        )
        comet_model = load_from_checkpoint(comet_path)
        accelerator.wait_for_everyone()  # Sync after loading

    # Quantization config
    quant_cfg = config.get("quantization", {})
    use_quantization = quant_cfg.get("load_in_4bit", False)

    if use_quantization:
        compute_dtype = getattr(
            torch, quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16")
        )
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
        )
    else:
        bnb_config = None

    # Load model
    accelerator.print(f"Loading base model: {config['base_model']}")
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }
    if use_quantization:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = {"": accelerator.local_process_index}

    base_model = AutoModelForCausalLM.from_pretrained(
        config["base_model"], **model_kwargs
    )

    if use_quantization:
        base_model = prepare_model_for_kbit_training(base_model)

    # Load adapter
    adapter_path = config.get("adapter_path")
    if adapter_path:
        accelerator.print(f"Loading adapter from {adapter_path}")
        model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            token=os.getenv("HUGGING_FACE_TOKEN"),
        )
        # Unfreeze LoRA params
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True
    else:
        model = base_model

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config["base_model"], trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model.enable_input_require_grads()
    if accelerator.is_main_process:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        accelerator.print(f"Trainable params: {trainable:,} / {total:,}")

    # Dataset
    dataset_cfg = config.get("dataset", {})
    train_cfg = config.get("training", {})

    train_dataset = GRPODataset(
        tokenizer=tokenizer,
        max_length=train_cfg.get("max_prompt_length", 256),
        hf_repo=dataset_cfg.get("hf_repo"),
        local_path=dataset_cfg.get("local_path"),
        max_samples=dataset_cfg.get("max_samples"),
    )

    collator = GRPOCollator(
        tokenizer, max_length=train_cfg.get("max_prompt_length", 256)
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get("per_device_batch_size", 2),
        shuffle=True,
        collate_fn=collator,
        num_workers=train_cfg.get("dataloader_num_workers", 2),
        pin_memory=True,
        drop_last=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(train_cfg.get("learning_rate", 5e-7)),
        weight_decay=0.01,
    )

    # Steps
    num_epochs = train_cfg.get("num_epochs", 1)
    num_update_steps_per_epoch = (
        len(train_dataloader) // accelerator.gradient_accumulation_steps
    )
    max_train_steps = train_cfg.get("max_steps") or (
        num_epochs * num_update_steps_per_epoch
    )
    warmup_steps = int(max_train_steps * train_cfg.get("warmup_ratio", 0.1))

    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )

    # Prepare
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # Reward combiner
    reward_combiner = RewardCombiner(
        comet_model=comet_model,
        use_comet=reward_cfg.get("use_comet", True),
        use_bleu=reward_cfg.get("use_bleu", False),
        use_numeric=reward_cfg.get("use_numeric", True),
        comet_weight=reward_cfg.get("comet_weight", 0.55),
        bleu_weight=reward_cfg.get("bleu_weight", 0.0),
        length_weight=reward_cfg.get("length_weight", 0.1),
        no_copy_weight=reward_cfg.get("no_copy_weight", 0.1),
        not_empty_weight=reward_cfg.get("not_empty_weight", 0.1),
        numeric_weight=reward_cfg.get("numeric_weight", 0.15),
    )

    # Training config
    num_generations = train_cfg.get("num_generations", 4)
    max_new_tokens = train_cfg.get("max_new_tokens", 128)
    temperature = train_cfg.get("temperature", 0.7)
    top_p = train_cfg.get("top_p", 0.9)
    beta = train_cfg.get("beta", 0.1)
    logging_steps = train_cfg.get("logging_steps", 10)
    save_steps = train_cfg.get("save_steps", 200)
    output_dir = config.get("output_dir", "outputs_grpo")

    accelerator.print(f"\n{'='*60}")
    accelerator.print("GRPO Training Configuration:")
    accelerator.print(f"  Dataset size: {len(train_dataset)}")
    accelerator.print(f"  Batch size: {train_cfg.get('per_device_batch_size', 2)}")
    accelerator.print(f"  Num generations per prompt: {num_generations}")
    accelerator.print(f"  Max train steps: {max_train_steps}")
    accelerator.print(f"  Temperature: {temperature}")
    accelerator.print(f"  Beta (KL coefficient): {beta}")
    accelerator.print(f"{'='*60}\n")

    # Training loop
    global_step = 0
    total_reward = 0
    reward_count = 0

    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            disable=not accelerator.is_main_process,
        )

        for batch in progress_bar:
            t_start = time.time()

            # Generate responses (disable grad checkpointing for fast KV cache)
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.gradient_checkpointing_disable()
            model.eval()
            with torch.no_grad():
                responses = generate_responses(
                    unwrapped,
                    tokenizer,
                    batch["input_ids"],
                    batch["attention_mask"],
                    num_generations=num_generations,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            model.train()
            unwrapped.gradient_checkpointing_enable()
            t_gen = time.time()

            # Flatten all responses for batched reward computation
            all_sources = []
            all_generations = []
            all_references = []
            response_counts = []  # Track how many responses per prompt

            for src, ref, resps in zip(batch["sources"], batch["references"], responses):
                all_sources.extend([src] * len(resps))
                all_generations.extend(resps)
                all_references.extend([ref] * len(resps))
                response_counts.append(len(resps))

            # Single batched reward call (MUCH faster than per-prompt)
            rewards_dict = reward_combiner(all_sources, all_generations, all_references)
            all_combined = rewards_dict["combined"]

            # Reshape back to per-prompt groups
            batch_rewards = []
            idx = 0
            for count in response_counts:
                batch_rewards.append(all_combined[idx:idx + count])
                idx += count

            # Track average reward
            total_reward += sum(all_combined)
            reward_count += len(all_combined)
            t_reward = time.time()

            # Compute GRPO loss
            with accelerator.accumulate(model):
                loss = compute_grpo_loss(
                    accelerator.unwrap_model(model),
                    tokenizer,
                    batch["prompts"],
                    responses,
                    batch_rewards,
                    accelerator.device,
                    beta=beta,
                )

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            t_loss = time.time()

            if accelerator.sync_gradients:
                global_step += 1

                if global_step >= max_train_steps:
                    break

                # Logging
                if global_step % logging_steps == 0:
                    avg_reward = total_reward / max(reward_count, 1)
                    lr = lr_scheduler.get_last_lr()[0]

                    # Timing breakdown
                    gen_time = t_gen - t_start
                    reward_time = t_reward - t_gen
                    loss_time = t_loss - t_reward
                    total_time = t_loss - t_start

                    if accelerator.is_main_process:
                        progress_bar.set_postfix({
                            "loss": f"{loss.item():.4f}",
                            "reward": f"{avg_reward:.4f}",
                            "gen": f"{gen_time:.1f}s",
                            "rew": f"{reward_time:.1f}s",
                        })

                        if use_wandb:
                            wandb.log({
                                "train/loss": loss.item(),
                                "train/avg_reward": avg_reward,
                                "train/learning_rate": lr,
                                "train/global_step": global_step,
                                "timing/generation_sec": gen_time,
                                "timing/reward_sec": reward_time,
                                "timing/loss_sec": loss_time,
                                "timing/total_sec": total_time,
                            })

                    total_reward = 0
                    reward_count = 0

                # Save checkpoint
                if global_step % save_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        save_path = f"{output_dir}/checkpoint-{global_step}"
                        os.makedirs(save_path, exist_ok=True)
                        unwrapped = accelerator.unwrap_model(model)
                        unwrapped.save_pretrained(save_path)
                        tokenizer.save_pretrained(save_path)
                        accelerator.print(f"Saved checkpoint to {save_path}")

        if global_step >= max_train_steps:
            break

    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_path = f"{output_dir}/final_model"
        os.makedirs(final_path, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        accelerator.print(f"Final model saved to {final_path}")

        # Push to hub
        hf_config = config.get("huggingface", {})
        if hf_config.get("push_best_model"):
            api = HfApi()
            try:
                api.create_repo(
                    repo_id=hf_config["repo_id"],
                    repo_type="model",
                    private=hf_config.get("private", True),
                    exist_ok=True,
                    token=os.getenv("HUGGING_FACE_TOKEN"),
                )
                api.upload_folder(
                    folder_path=final_path,
                    repo_id=hf_config["repo_id"],
                    repo_type="model",
                    commit_message="GRPO trained model",
                    token=os.getenv("HUGGING_FACE_TOKEN"),
                )
                accelerator.print(f"Pushed to {hf_config['repo_id']}")
            except Exception as e:
                accelerator.print(f"Push failed: {e}")

    if use_wandb:
        wandb.finish()

    accelerator.print("\nGRPO Training complete!")


def main():
    parser = argparse.ArgumentParser(description="GRPO Training for Translation")
    parser.add_argument(
        "--config",
        type=str,
        default="config_grpo.yaml",
        help="Path to config YAML file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    os.makedirs(config.get("output_dir", "outputs_grpo"), exist_ok=True)

    train(config)


if __name__ == "__main__":
    main()
