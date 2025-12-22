"""
Sample and score training data to create gold set for GRPO.

Workflow:
1. Sample N examples from train.csv
2. Run model inference to get predictions
3. Score with COMET
4. Bin by COMET scores (70% good, 20% medium, 10% bad)
5. Export each bin to separate files for manual review

Usage:
    python sample_gold_set.py --sample_size 10000 --output_dir grpo_gold
"""
import argparse
import gc
import os

import pandas as pd
import torch
from comet import download_model, load_from_checkpoint
from datasets import load_dataset
from dotenv import load_dotenv
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()


def parse_tag(text: str) -> str:
    """Extract language tag from source text."""
    if text.startswith("[VI] "):
        return "VI"
    elif text.startswith("[EN] "):
        return "EN"
    return "UNK"


def clean_source(text: str) -> str:
    """Remove language tag from source text."""
    if text.startswith("[VI] "):
        return text[5:]
    elif text.startswith("[EN] "):
        return text[5:]
    return text


def batch_generate(model, tokenizer, prompts: list, max_new_tokens: int = 256) -> list:
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
    input_len = inputs["input_ids"].shape[1]

    for output in outputs:
        generated = output[input_len:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        # Clean any language tags from output
        if text.startswith("[VI] "):
            text = text[5:]
        elif text.startswith("[EN] "):
            text = text[5:]
        predictions.append(text)

    return predictions


def run_inference(
    df: pd.DataFrame,
    base_model_path: str,
    adapter_path: str,
    batch_size: int = 16,
    max_new_tokens: int = 256,
) -> list:
    """Run model inference on dataframe."""
    print(f"Loading base model: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(
        model,
        adapter_path,
        token=os.getenv("HUGGING_FACE_TOKEN"),
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model.eval()

    # Prepare prompts - use src directly (it has the language tag)
    prompts = [row["src"] + " " for _, row in df.iterrows()]

    predictions = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i : i + batch_size]
        batch_preds = batch_generate(model, tokenizer, batch_prompts, max_new_tokens)
        predictions.extend(batch_preds)

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return predictions


def score_with_comet(
    df: pd.DataFrame,
    comet_model_name: str = "Unbabel/wmt22-comet-da",
    batch_size: int = 64,
) -> list:
    """Score predictions with COMET."""
    print(f"Loading COMET model: {comet_model_name}")
    comet_path = download_model(comet_model_name)
    comet_model = load_from_checkpoint(comet_path)

    # Prepare COMET input
    comet_data = []
    for _, row in df.iterrows():
        comet_data.append({
            "src": clean_source(row["src"]),
            "mt": row["prediction"],
            "ref": row["tgt"],
        })

    print("Computing COMET scores...")
    output = comet_model.predict(comet_data, batch_size=batch_size, gpus=1)

    return output.scores


def bin_by_comet(df: pd.DataFrame, output_dir: str):
    """Bin samples by COMET score and export to separate files."""
    os.makedirs(output_dir, exist_ok=True)

    # Define bins: 70% good (>=0.8), 20% medium (0.6-0.8), 10% bad (<0.6)
    bins = [
        ("00_very_bad", 0.0, 0.6),
        ("01_bad", 0.6, 0.7),
        ("02_medium", 0.7, 0.8),
        ("03_good", 0.8, 0.9),
        ("04_excellent", 0.9, 1.0),
    ]

    print("\nCOMET Score Distribution:")
    print("-" * 40)

    for bin_name, lo, hi in bins:
        if bin_name == "04_excellent":
            subset = df[(df["comet"] >= lo) & (df["comet"] <= hi)]
        else:
            subset = df[(df["comet"] >= lo) & (df["comet"] < hi)]

        pct = len(subset) / len(df) * 100
        print(f"{lo:.1f}–{hi:.1f}: {len(subset):5d} ({pct:5.1f}%)")

        if len(subset) > 0:
            # Sort by comet score ascending (worst first for review)
            subset_sorted = subset.sort_values("comet", ascending=True)
            output_path = os.path.join(output_dir, f"{bin_name}.csv")
            subset_sorted.to_csv(output_path, index=False, encoding="utf-8")
            print(f"  -> Saved to {output_path}")

    print("-" * 40)

    # Also save full dataset with scores
    full_path = os.path.join(output_dir, "all_scored.csv")
    df.to_csv(full_path, index=False, encoding="utf-8")
    print(f"\nFull dataset saved to {full_path}")

    # Print summary stats
    print(f"\nSummary Statistics:")
    print(df["comet"].describe())


def main():
    parser = argparse.ArgumentParser(description="Sample gold set for GRPO")
    parser.add_argument(
        "--hf_repo",
        type=str,
        default="TranDuong/medical-vlsp-2025",
        help="HuggingFace dataset repo",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=10000,
        help="Number of samples to evaluate",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Base model path",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default="TranDuong/qwen-medical-mt-cpt",
        help="Adapter model path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="train_grpo/grpo_gold",
        help="Output directory for binned files",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--skip_inference",
        action="store_true",
        help="Skip inference, load from existing scored file",
    )
    args = parser.parse_args()

    if args.skip_inference:
        # Load existing scored data
        scored_path = os.path.join(args.output_dir, "all_scored.csv")
        print(f"Loading existing scored data from {scored_path}")
        df = pd.read_csv(scored_path)
    else:
        # Load from HuggingFace
        print(f"Loading training data from HuggingFace: {args.hf_repo}")
        dataset = load_dataset(
            args.hf_repo,
            data_files={"train": "cleaned_data/train.csv"},
            split="train",
            token=os.getenv("HUGGING_FACE_TOKEN"),
        )
        train_df = dataset.to_pandas()
        print(f"Total training samples: {len(train_df)}")

        # Stratified sampling by language tag
        train_df["tag"] = train_df["src"].apply(parse_tag)

        # Sample proportionally from each direction
        samples = []
        for tag in ["VI", "EN"]:
            tag_df = train_df[train_df["tag"] == tag]
            n = min(args.sample_size // 2, len(tag_df))
            samples.append(tag_df.sample(n=n, random_state=args.seed))

        df = pd.concat(samples, ignore_index=True)
        df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
        print(f"Sampled {len(df)} examples (balanced EN/VI)")

        # Run inference
        predictions = run_inference(
            df,
            args.base_model,
            args.adapter,
            batch_size=args.batch_size,
        )
        df["prediction"] = predictions

        # Score with COMET
        scores = score_with_comet(df)
        df["comet"] = scores

    # Bin and export
    bin_by_comet(df, args.output_dir)

    print("\nDone!")
    print("\nNext steps:")
    print("1. Review 00_very_bad.csv and 01_bad.csv - fix or remove bad translations")
    print("2. Spot check 02_medium.csv for edge cases")
    print("3. Use 03_good.csv and 04_excellent.csv as positive examples")


if __name__ == "__main__":
    main()
