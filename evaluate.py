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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
    for i, output in enumerate(outputs):
        input_len = inputs["attention_mask"][i].sum().item()
        generated = output[input_len:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        text = clean_prediction(text)
        predictions.append(text.strip())

    return predictions


def clean_prediction(text: str) -> str:
    """Clean prediction to extract only the target translation."""
    import re

    if "[VI]" in text and "[EN]" in text:
        vi_pos = text.rfind("[VI]")
        en_pos = text.rfind("[EN]")
        if vi_pos > en_pos:
            text = text.split("[VI]")[-1].strip()
        else:
            text = text.split("[EN]")[-1].strip()
    elif "[VI]" in text:
        text = text.split("[VI]")[-1].strip()
    elif "[EN]" in text:
        text = text.split("[EN]")[-1].strip()

    text = re.sub(r'(\d+\.\s*){3,}.*$', '', text)
    text = re.sub(r'[\s\d\.]+$', '', text)

    return text.strip()


def compute_perplexity(model, tokenizer, dataset, max_samples: int = 1000) -> float:
    """Compute perplexity on the dataset."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    samples = list(dataset.take(max_samples))

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
    num_examples: int,
) -> list:
    """Save random examples to JSON."""
    indices = list(range(len(sources)))
    if len(indices) > num_examples:
        random.seed(42)
        indices = random.sample(indices, num_examples)

    examples = []
    for idx in indices:
        examples.append({
            "source": sources[idx],
            "prediction": predictions[idx],
            "reference": references[idx],
        })

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
    # Quantization config
    quant_cfg = config.get("quantization", {})
    compute_dtype = getattr(torch, quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16"))
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
    )

    # Load model
    print(f"Loading base model: {config['base_model']}")
    base_model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading adapter from {adapter_model_path}")
    hf_token = os.getenv("HUGGING_FACE_TOKEN")
    print(f"Using HF token: {hf_token[:10]}..." if hf_token else "WARNING: No HF token found!")
    model = PeftModel.from_pretrained(base_model, adapter_model_path, token=hf_token)
    # Load tokenizer from base model (same tokenizer, avoids auth issues with private adapter repo)
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    model.eval()

    # Load test dataset from HuggingFace
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
        streaming=True,
        token=os.getenv("HUGGING_FACE_TOKEN"),
    )

    # Compute perplexity first
    print("\nComputing perplexity...")
    perplexity = compute_perplexity(model, tokenizer, test_dataset, max_samples=1000)
    print(f"Perplexity: {perplexity:.2f}")

    # Reload dataset for translation evaluation
    test_dataset = load_dataset(
        hf_repo,
        data_files={test_split: split_to_file.get(test_split, f"cleaned_data/{test_split}.csv")},
        split=test_split,
        streaming=True,
        token=os.getenv("HUGGING_FACE_TOKEN"),
    )

    # Parse test data
    sources = []
    prompts = []
    references = []

    eval_cfg = config.get("evaluation", {})
    max_samples = eval_cfg.get("max_samples", 2000)

    for example in tqdm(test_dataset.take(max_samples), desc="Loading test data"):
        src = str(example.get("src", "")).strip()
        tgt = str(example.get("tgt", "")).strip()

        if not src or not tgt:
            continue

        # src already contains [EN] or [VI] prefix
        sources.append(src)
        prompts.append(src + " ")  # Add space for generation
        references.append(tgt)

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

    # Save examples
    if eval_cfg.get("log_examples", True):
        num_examples = eval_cfg.get("num_examples_to_log", 100)
        examples_path = f"{output_dir}/eval_examples.json"
        save_examples(sources, predictions, references, examples_path, num_examples)

    # Compute metrics
    print("\nComputing BLEU...")
    bleu = corpus_bleu(predictions, [references])

    print("Computing chrF++...")
    chrf = corpus_chrf(predictions, [references], word_order=2)

    print("Computing COMET...")
    comet_model_path = download_model(eval_cfg.get("comet_model", "Unbabel/wmt22-comet-da"))
    comet_model = load_from_checkpoint(comet_model_path)

    # Clean sources for COMET (remove language tags)
    clean_sources = []
    for s in sources:
        if s.startswith("[EN]"):
            clean_sources.append(s[4:].strip())
        elif s.startswith("[VI]"):
            clean_sources.append(s[4:].strip())
        else:
            clean_sources.append(s)

    comet_data = [
        {"src": s, "mt": p, "ref": r}
        for s, p, r in zip(clean_sources, predictions, references)
    ]
    comet_score = comet_model.predict(
        comet_data,
        batch_size=eval_cfg.get("comet_batch_size", 32),
        gpus=1,
    ).system_score

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
