# orchestrator.py
"""
Training orchestrator for curriculum learning with Qwen finetuning.
Manages multi-part training, evaluation, early stopping, and model upload.
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

import wandb
import yaml
from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download

load_dotenv()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def download_dataset(repo_id: str, local_dir: str) -> None:
    """Download dataset from HuggingFace Hub if not exists."""
    if Path(local_dir).exists():
        print(f"Dataset already exists at {local_dir}")
        return

    print(f"Downloading dataset from {repo_id}...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        token=os.getenv("HUGGING_FACE_TOKEN"),
    )
    print(f"Dataset downloaded to {local_dir}")


class TrainingOrchestrator:
    def __init__(self, config: dict):
        self.config = config
        self.best_bleu = 0.0
        self.best_checkpoint = None
        self.history = []
        self.wandb_run = None

        # Download dataset
        download_dataset(
            config["dataset"]["hf_repo"],
            config["dataset"]["local_dir"],
        )

    def _init_wandb(self) -> None:
        """Initialize single W&B run for evaluation metrics (line charts)."""
        if not self.config["wandb"]["enabled"]:
            return

        wandb.login(key=os.getenv("WANDB_API_KEY"))
        self.wandb_run = wandb.init(
            project=self.config["wandb"]["project"],
            name=f"{self.config['experiment_name']}-eval",
            config=self.config,
            group=self.config["experiment_name"],
            job_type="eval",
            tags=["evaluation", "metrics"],
        )

    def train_part(self, part_idx: int, adapter_path: str = None) -> str:
        """Train a single curriculum part."""
        part_file = f"train_part{part_idx + 1}.txt"
        train_path = f"{self.config['dataset']['local_dir']}/train/{part_file}"
        valid_path = f"{self.config['dataset']['local_dir']}/val/val.txt"

        run_name = f"{self.config['experiment_name']}-part{part_idx + 1}"
        output_dir = f"{self.config['output_dir']}/{run_name}"

        # Build command
        cmd = ["accelerate", "launch"]
        num_gpus = self.config["training"]["num_gpus"]
        if num_gpus > 1:
            cmd.extend(["--multi_gpu", "--num_processes", str(num_gpus)])
        else:
            cmd.extend(["--num_processes", "1"])

        cmd.extend([
            "train.py",
            "--config", "config.yaml",
            "--train_data_path", train_path,
            "--valid_data_path", valid_path,
            "--output_dir", output_dir,
            "--run_name", run_name,
            "--part", str(part_idx + 1),
        ])
        if adapter_path:
            cmd.extend(["--adapter_model_path", adapter_path])

        print(f"Training: {run_name}")
        subprocess.run(cmd, check=True)
        return f"{output_dir}/best_model"

    def evaluate(self, adapter_path: str, part_idx: int) -> dict:
        """Evaluate model on test set."""
        test_path = f"{self.config['dataset']['local_dir']}/test/test_all.txt"

        cmd = [
            "python", "evaluate.py",
            "--config", "config.yaml",
            "--adapter_model_path", adapter_path,
            "--test_data_path", test_path,
            "--part", str(part_idx + 1),
            "--output_dir", self.config["output_dir"],
        ]

        print(f"Evaluating part {part_idx + 1}...")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Parse JSON output from evaluate.py
        metrics = json.loads(result.stdout.strip().split("\n")[-1])
        return metrics

    def log_examples_to_wandb(self, part_idx: int) -> None:
        """Log examples from JSON to W&B as a table."""
        if not self.wandb_run:
            return

        examples_path = f"{self.config['output_dir']}/part-{part_idx + 1}-examples.json"
        if not os.path.exists(examples_path):
            return

        with open(examples_path, "r", encoding="utf-8") as f:
            examples = json.load(f)

        # Create W&B table
        table = wandb.Table(columns=["Source", "Prediction", "Reference"])
        for ex in examples:
            table.add_data(ex["source"], ex["prediction"], ex["reference"])

        wandb.log({f"examples/part-{part_idx + 1}": table})
        print(f"Logged {len(examples)} examples to W&B")

    def push_to_hub(self) -> None:
        """Push best model adapter to HuggingFace Hub."""
        if not self.config["huggingface"]["push_best_model"]:
            return

        if not self.best_checkpoint:
            print("No best checkpoint to push")
            return

        print(f"Pushing best model to HuggingFace Hub...")
        api = HfApi()

        repo_id = self.config["huggingface"]["repo_id"]
        private = self.config["huggingface"]["private"]

        # Create repo if not exists
        try:
            api.create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=private,
                exist_ok=True,
                token=os.getenv("HUGGING_FACE_TOKEN"),
            )
        except Exception as e:
            print(f"Repo creation note: {e}")

        # Upload adapter folder
        api.upload_folder(
            folder_path=self.best_checkpoint,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Best model - BLEU: {self.best_bleu:.2f}",
            token=os.getenv("HUGGING_FACE_TOKEN"),
        )
        print(f"Model pushed to https://huggingface.co/{repo_id}")

    def save_progress(self) -> None:
        """Save training progress to JSON file."""
        progress = {
            "history": self.history,
            "best_bleu": self.best_bleu,
            "best_checkpoint": self.best_checkpoint,
            "config": self.config,
        }
        progress_path = f"{self.config['output_dir']}/progress.json"
        with open(progress_path, "w") as f:
            json.dump(progress, f, indent=2)

    def run_pipeline(self) -> None:
        """Run full curriculum training pipeline."""
        # Initialize single W&B run for eval metrics (line charts)
        self._init_wandb()

        adapter_path = None
        no_improve = 0
        num_parts = self.config["dataset"]["num_parts"]
        patience = self.config["early_stopping"]["patience"]
        min_delta = self.config["early_stopping"]["min_delta"]

        try:
            for i in range(num_parts):
                print(f"\n{'=' * 60}")
                print(f"Part {i + 1}/{num_parts}")
                print(f"{'=' * 60}")

                # Train (creates its own W&B run with loss graphs)
                adapter_path = self.train_part(i, adapter_path)

                # Evaluate (returns metrics, saves examples to JSON)
                metrics = self.evaluate(adapter_path, i)
                bleu = metrics["bleu"]

                # Log examples to W&B table
                self.log_examples_to_wandb(i)

                self.history.append({
                    "part": i + 1,
                    "metrics": metrics,
                    "checkpoint": adapter_path,
                })

                print(f"Results - BLEU: {bleu:.2f}, chrF++: {metrics['chrf++']:.2f}, COMET: {metrics['comet']:.4f}")

                # Log eval metrics to orchestrator's W&B run (line charts)
                if self.wandb_run:
                    wandb.log({
                        "stage": i + 1,
                        "eval/bleu": bleu,
                        "eval/chrf++": metrics["chrf++"],
                        "eval/comet": metrics["comet"],
                    })

                # Early stopping check
                if bleu > self.best_bleu + min_delta:
                    self.best_bleu = bleu
                    self.best_checkpoint = adapter_path
                    no_improve = 0
                    print(f"New best BLEU: {bleu:.2f}")
                else:
                    no_improve += 1
                    print(f"No improvement ({no_improve}/{patience})")

                self.save_progress()

                if no_improve >= patience:
                    print(f"\nEarly stopping triggered at part {i + 1}")
                    break

            print(f"\n{'=' * 60}")
            print("Training complete!")
            print(f"Best BLEU: {self.best_bleu:.2f}")
            print(f"Best checkpoint: {self.best_checkpoint}")
            print(f"{'=' * 60}")

            # Push best model to HuggingFace
            self.push_to_hub()

        finally:
            if self.wandb_run:
                wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Qwen Finetuning Orchestrator")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config YAML file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)

    orchestrator = TrainingOrchestrator(config)
    orchestrator.run_pipeline()


if __name__ == "__main__":
    main()
