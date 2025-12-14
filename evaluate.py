# evaluate.py
"""
Evaluation script for translation quality metrics.
Computes BLEU, chrF++, and COMET scores with batched generation.
Logs examples to JSON and W&B table.
"""
import argparse
import json
import os
import random

import torch
import yaml
from comet import download_model, load_from_checkpoint
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
    # Tokenize with padding
    tokenizer.padding_side = "left"  # For decoder-only models
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
        )

    # Decode outputs
    predictions = []
    for i, output in enumerate(outputs):
        # Remove input tokens from output
        input_len = inputs["attention_mask"][i].sum().item()
        generated = output[input_len:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        predictions.append(text.strip())

    return predictions


def save_examples(
    sources: list,
    predictions: list,
    references: list,
    output_path: str,
    num_examples: int,
    config: dict,
    part: int,
) -> list:
    """Save random examples to JSON and return sampled data for W&B."""
    # Sample random indices
    indices = list(range(len(sources)))
    if len(indices) > num_examples:
        random.seed(42)  # Reproducible sampling
        indices = random.sample(indices, num_examples)

    examples = []
    for idx in indices:
        examples.append({
            "source": sources[idx],
            "prediction": predictions[idx],
            "reference": references[idx],
        })

    # Save to JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(examples)} examples to {output_path}")

    return examples


def evaluate(
    config: dict,
    adapter_model_path: str,
    test_data_path: str,
    part: int,
    output_dir: str,
) -> dict:
    """Evaluate model on test set and return metrics.

    Note: W&B logging is handled by orchestrator for unified eval run.
    This function just computes metrics and saves examples to JSON.
    """
    # Quantization config
    quant_cfg = config["quantization"]
    compute_dtype = getattr(torch, quant_cfg["bnb_4bit_compute_dtype"])
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
    )

    # Load model
    base_model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, adapter_model_path)
    tokenizer = AutoTokenizer.from_pretrained(adapter_model_path)
    model.eval()

    # Load test data
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_lines = f.readlines()

    # Apply sample limit if configured
    max_samples = config["dataset"]["max_test_samples"]
    if max_samples:
        test_lines = test_lines[:max_samples]
        print(f"Limited to {len(test_lines)} test samples")

    # Parse test data
    sources = []
    prompts = []
    references = []

    for line in test_lines:
        parts = line.strip().split("[VI]")
        prompt = parts[0] + "[VI]"
        reference = parts[1].strip() if len(parts) > 1 else ""

        sources.append(parts[0].replace("[EN]", "").strip())
        prompts.append(prompt)
        references.append(reference)

    # Batched generation
    eval_cfg = config["evaluation"]
    batch_size = eval_cfg.get("generation_batch_size", 8)
    predictions = []

    for i in tqdm(range(0, len(prompts), batch_size), desc=f"Evaluating part {part}"):
        batch_prompts = prompts[i : i + batch_size]
        batch_preds = batch_generate(
            model, tokenizer, batch_prompts, eval_cfg["max_new_tokens"]
        )
        predictions.extend(batch_preds)

    # Save examples to JSON
    if eval_cfg.get("log_examples", True):
        num_examples = eval_cfg.get("num_examples_to_log", 100)
        examples_path = f"{output_dir}/part-{part}-examples.json"
        save_examples(
            sources, predictions, references, examples_path, num_examples, config, part
        )

    # BLEU
    bleu = corpus_bleu(predictions, [references])

    # chrF++
    chrf = corpus_chrf(predictions, [references], word_order=2)

    # COMET
    comet_model_path = download_model(eval_cfg["comet_model"])
    comet_model = load_from_checkpoint(comet_model_path)
    comet_data = [
        {"src": s, "mt": p, "ref": r}
        for s, p, r in zip(sources, predictions, references)
    ]
    comet_score = comet_model.predict(
        comet_data,
        batch_size=eval_cfg["comet_batch_size"],
        gpus=1,
    ).system_score

    results = {
        "bleu": bleu.score,
        "chrf++": chrf.score,
        "comet": comet_score,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Qwen Finetuning Evaluator")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config YAML file"
    )
    parser.add_argument(
        "--adapter_model_path", type=str, required=True, help="Path to adapter model"
    )
    parser.add_argument(
        "--test_data_path", type=str, required=True, help="Path to test data"
    )
    parser.add_argument(
        "--part", type=int, required=True, help="Curriculum part number"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Output directory for examples"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    results = evaluate(
        config=config,
        adapter_model_path=args.adapter_model_path,
        test_data_path=args.test_data_path,
        part=args.part,
        output_dir=args.output_dir,
    )

    # Output JSON for orchestrator to parse
    print(json.dumps(results))


if __name__ == "__main__":
    main()
