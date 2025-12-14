# evaluate.py
import argparse
import json
import os
import torch
import wandb
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sacrebleu import corpus_bleu, corpus_chrf
from comet import download_model, load_from_checkpoint
from tqdm import tqdm

load_dotenv()


def evaluate(
    base_model_path,
    adapter_model_path,
    test_data_path,
    run_name,
    wandb_project,
):
    # W&B init
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(project=wandb_project, name=run_name)

    # Load model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, adapter_model_path)
    tokenizer = AutoTokenizer.from_pretrained(adapter_model_path)
    model.eval()

    # Load test data
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_lines = f.readlines()

    # Generate translations
    sources, predictions, references = [], [], []

    for line in tqdm(test_lines, desc="Generating translations"):
        parts = line.strip().split("[VI]")
        source = parts[0] + "[VI]"
        reference = parts[1].strip() if len(parts) > 1 else ""

        inputs = tokenizer(source, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction = prediction.split("[VI]")[-1].strip()

        sources.append(parts[0].replace("[EN]", "").strip())
        predictions.append(prediction)
        references.append(reference)

    # BLEU
    bleu = corpus_bleu(predictions, [references])

    # chrF++
    chrf = corpus_chrf(predictions, [references], word_order=2)

    # COMET
    comet_model_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(comet_model_path)
    comet_data = [
        {"src": s, "mt": p, "ref": r}
        for s, p, r in zip(sources, predictions, references)
    ]
    comet_score = comet_model.predict(comet_data, batch_size=8, gpus=1).system_score

    results = {
        "bleu": bleu.score,
        "chrf++": chrf.score,
        "comet": comet_score,
    }

    # Log to W&B
    wandb.log(results)
    wandb.finish()

    # Output JSON for orchestrator
    print(json.dumps(results))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--adapter_model_path", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--wandb_project", type=str, required=True)
    parser.add_argument("--max_test_samples", type=int, default=None)
    args = parser.parse_args()

    # Patch: limit test samples if requested
    import builtins
    orig_open = builtins.open
    def limited_open(file, mode='r', encoding=None, *a, **kw):
        if file == args.test_data_path and 'r' in mode and args.max_test_samples is not None:
            with orig_open(file, mode, encoding=encoding, *a, **kw) as f:
                lines = f.readlines()[:args.max_test_samples]
            import io
            return io.StringIO(''.join(lines))
        return orig_open(file, mode, encoding=encoding, *a, **kw)
    builtins.open = limited_open

    evaluate(
        args.base_model_path,
        args.adapter_model_path,
        args.test_data_path,
        args.run_name,
        args.wandb_project,
    )