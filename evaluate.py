"""
Evaluation script for translation quality metrics.
Computes BLEU, chrF++, COMET, and perplexity scores with batched generation.
"""
import argparse
import json
import math
import os
import random

import torch
import yaml
from comet import download_model, load_from_checkpoint
from datasets import load_dataset
from dotenv import load_dotenv
from peft import PeftModel
from sacrebleu import corpus_bleu, corpus_chrf
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def batch_generate(model, tokenizer, prompts: list, max_new_tokens: int) -> list:
    """Generate translations for a batch of prompts."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
        )

    predictions = []
    # With left-padding, the input tokens are at the END of the padded sequence
    # So generated tokens start after the full padded input length
    input_len = inputs["input_ids"].shape[1]

    for i, output in enumerate(outputs):
        generated = output[input_len:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        text = clean_prediction(text)
        predictions.append(text.strip())

    return predictions


def clean_prediction(text: str) -> str:
    """Clean prediction to extract only the target translation."""
    text = text.strip()

    # Remove language tags if present
    if text.startswith("[VI] "):
        text = text[5:]
    elif text.startswith("[EN] "):
        text = text[5:]

    return text.strip()


def compute_perplexity(model, tokenizer, dataset, max_samples: int = 1000) -> float:
    """Compute perplexity on the dataset."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    samples = dataset.select(range(min(max_samples, len(dataset))))

    for example in tqdm(samples, desc="Computing perplexity"):
        src = str(example.get("src", "")).strip()
        tgt = str(example.get("tgt", "")).strip()

        if not src or not tgt:
            continue

        # Format: [lang] source target
        text = f"{src} {tgt}{tokenizer.eos_token}"

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()

        num_tokens = inputs["attention_mask"].sum().item()
        total_loss += loss * num_tokens
        total_tokens += num_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = math.exp(avg_loss)

    return perplexity


def save_examples(
    sources: list,
    predictions: list,
    references: list,
    output_path: str,
    num_examples: int = None,
    metrics: dict = None,
) -> list:
    """Save examples to JSON. If num_examples is None or -1, save all.

    Args:
        metrics: Optional dict of metric_name -> list of per-sample scores.
                 e.g., {"comet": [0.85, 0.92, ...]}
    """
    indices = list(range(len(sources)))
    if num_examples and num_examples > 0 and len(indices) > num_examples:
        random.seed(42)
        indices = random.sample(indices, num_examples)

    examples = []
    for idx in indices:
        example = {
            "source": sources[idx],
            "prediction": predictions[idx],
            "reference": references[idx],
        }
        # Add per-sample metrics if available
        if metrics:
            for metric_name, scores in metrics.items():
                if scores and idx < len(scores):
                    example[metric_name] = scores[idx]
        examples.append(example)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(examples)} examples to {output_path}")

    return examples


def evaluate(
    config: dict,
    adapter_model_path: str,
    output_dir: str,
    test_split: str = "test",
) -> dict:
    """Evaluate model on test set and return metrics."""
    # Load model
    print(f"Loading base model: {config['base_model']}")
    base_model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading adapter from {adapter_model_path}")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_model_path,
        token=os.getenv("HUGGING_FACE_TOKEN")
    )
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    model.eval()

    # Load test dataset
    dataset_cfg = config.get("dataset", {})
    hf_repo = dataset_cfg.get("hf_repo", "TranDuong/medical-vlsp-2025")

    print(f"Loading test data from {hf_repo}")
    split_to_file = {
        "train": "cleaned_data/train.csv",
        "validation": "cleaned_data/val.csv",
        "test": "cleaned_data/test.csv",
    }
    test_dataset = load_dataset(
        hf_repo,
        data_files={test_split: split_to_file.get(test_split, f"cleaned_data/{test_split}.csv")},
        split=test_split,
        streaming=False,
        token=os.getenv("HUGGING_FACE_TOKEN"),
    )

    # Compute perplexity
    print("\nComputing perplexity...")
    perplexity = compute_perplexity(model, tokenizer, test_dataset, max_samples=1000)
    print(f"Perplexity: {perplexity:.2f}")

    # Parse test data
    eval_cfg = config.get("evaluation", {})
    max_samples = eval_cfg.get("max_samples", 2000)

    test_samples = test_dataset.select(range(min(max_samples, len(test_dataset))))

    sources = []
    prompts = []
    references = []

    for example in tqdm(test_samples, desc="Loading test data"):
        src = str(example.get("src", "")).strip()
        tgt = str(example.get("tgt", "")).strip()

        if not src or not tgt:
            continue

        sources.append(src)
        prompts.append(src + " ")

        # Clean reference - remove language tags if present
        clean_ref = tgt
        if tgt.startswith("[VI] "):
            clean_ref = tgt[5:]
        elif tgt.startswith("[EN] "):
            clean_ref = tgt[5:]
        references.append(clean_ref)

    print(f"Loaded {len(sources)} test samples")

    # Batched generation
    batch_size = eval_cfg.get("generation_batch_size", 32)
    predictions = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating translations"):
        batch_prompts = prompts[i:i + batch_size]
        batch_preds = batch_generate(
            model, tokenizer, batch_prompts, eval_cfg.get("max_new_tokens", 256)
        )
        predictions.extend(batch_preds)

    # Compute metrics
    print("\nComputing BLEU...")
    bleu = corpus_bleu(predictions, [references])

    print("Computing chrF++...")
    chrf = corpus_chrf(predictions, [references], word_order=2)

    print("Computing COMET...")
    comet_model_path = download_model(eval_cfg.get("comet_model", "Unbabel/wmt22-comet-da"))
    comet_model = load_from_checkpoint(comet_model_path)

    comet_data = [
        {
            "src": s[5:] if s.startswith("[EN] ") or s.startswith("[VI] ") else s,
            "mt": p,
            "ref": r
        }
        for s, p, r in zip(sources, predictions, references)
    ]
    comet_output = comet_model.predict(
        comet_data,
        batch_size=eval_cfg.get("comet_batch_size", 32),
        gpus=1,
    )
    comet_score = comet_output.system_score
    comet_scores = comet_output.scores  # Per-sample scores

    # Save examples with per-sample metrics
    if eval_cfg.get("log_examples", True):
        num_examples = eval_cfg.get("num_examples_to_log")  # None = all
        examples_path = f"{output_dir}/eval_examples.json"
        per_sample_metrics = {"comet": comet_scores}
        save_examples(
            sources, predictions, references, examples_path,
            num_examples, metrics=per_sample_metrics
        )

    results = {
        "bleu": bleu.score,
        "chrf++": chrf.score,
        "comet": comet_score,
        "perplexity": perplexity,
    }

    print("\n" + "=" * 50)
    print("Evaluation Results:")
    print(f"  BLEU:       {results['bleu']:.2f}")
    print(f"  chrF++:     {results['chrf++']:.2f}")
    print(f"  COMET:      {results['comet']:.4f}")
    print(f"  Perplexity: {results['perplexity']:.2f}")
    print("=" * 50)

    return results


def main():
    parser = argparse.ArgumentParser(description="Qwen Translation Evaluator")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--adapter_model_path",
        type=str,
        required=True,
        help="Path to adapter model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory for examples",
    )
    parser.add_argument(
        "--test_split",
        type=str,
        default="test",
        help="Test split name (default: test)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    results = evaluate(
        config=config,
        adapter_model_path=args.adapter_model_path,
        output_dir=args.output_dir,
        test_split=args.test_split,
    )

    # Output JSON for scripting
    print(json.dumps(results))


if __name__ == "__main__":
    main()
