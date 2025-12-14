# orchestrator.py
import subprocess
import json
from pathlib import Path
from dotenv import load_dotenv
import os
from huggingface_hub import snapshot_download

load_dotenv()


def download_dataset(repo_id, local_dir):
    """Download dataset from HuggingFace Hub"""
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
    def __init__(self, config):
        self.config = config
        self.best_bleu = 0
        self.best_checkpoint = None
        self.history = []
        
        # Download dataset
        download_dataset(config["hf_dataset"], config["dataset_dir"])
        
    def train_part(self, part_idx, adapter_path=None):
        part_file = f"train_part{part_idx + 1}.txt"
        train_path = f"{self.config['dataset_dir']}/train/{part_file}"
        
        run_name = f"{self.config['model_name']}-part{part_idx + 1}"
        output_dir = f"{self.config['output_dir']}/{run_name}"
        
        cmd = ["accelerate", "launch"]
        if self.config["num_gpus"] > 1:
            cmd.extend(["--multi_gpu", "--num_processes", str(self.config["num_gpus"])] )
        else:
            cmd.extend(["--num_processes", "1"])
        cmd.extend([
            "train.py",
            "--base_model_path", self.config["base_model"],
            "--train_data_path", train_path,
            "--valid_data_path", f"{self.config['dataset_dir']}/val/val.txt",
            "--output_dir", output_dir,
            "--run_name", run_name,
            "--wandb_project", self.config["wandb_project"],
        ])
        if adapter_path:
            cmd.extend(["--adapter_model_path", adapter_path])
        
        print(f"Training: {run_name}")
        subprocess.run(cmd, check=True)
        return f"{output_dir}/best_model"
    
    def evaluate(self, adapter_path, part_idx):
        run_name = f"{self.config['model_name']}-part{part_idx + 1}-eval"
        # Limit test set for smoke test
        result = subprocess.run([
            "python", "evaluate.py",
            "--base_model_path", self.config["base_model"],
            "--adapter_model_path", adapter_path,
            "--test_data_path", f"{self.config['dataset_dir']}/test/test_all.txt",
            "--run_name", run_name,
            "--wandb_project", self.config["wandb_project"],
            "--max_test_samples", "10",
        ], capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    
    def run_pipeline(self):
        adapter_path = None
        no_improve = 0
        
        for i in range(self.config["num_parts"]):
            print(f"\n{'='*60}")
            print(f"Part {i + 1}/{self.config['num_parts']}")
            print(f"{'='*60}")
            
            # Train
            adapter_path = self.train_part(i, adapter_path)
            
            # Evaluate
            metrics = self.evaluate(adapter_path, i)
            bleu = metrics["bleu"]
            
            self.history.append({
                "part": i + 1,
                "metrics": metrics,
                "checkpoint": adapter_path
            })
            
            print(f"Results - BLEU: {bleu:.2f}, chrF++: {metrics['chrf++']:.2f}, COMET: {metrics['comet']:.4f}")
            
            # Early stopping
            if bleu > self.best_bleu + self.config["min_delta"]:
                self.best_bleu = bleu
                self.best_checkpoint = adapter_path
                no_improve = 0
                print(f"! New best BLEU: {bleu:.2f}")
            else:
                no_improve += 1
                print(f"!!! No improvement ({no_improve}/{self.config['patience']})")
            
            self.save_progress()
            
            if no_improve >= self.config["patience"]:
                print(f"\nEarly stopping triggered at part {i + 1}")
                break
        
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Best BLEU: {self.best_bleu:.2f}")
        print(f"Best checkpoint: {self.best_checkpoint}")
        print(f"{'='*60}")
        
    def save_progress(self):
        progress = {
            "history": self.history,
            "best_bleu": self.best_bleu,
            "best_checkpoint": self.best_checkpoint,
            "config": self.config,
        }
        with open(f"{self.config['output_dir']}/progress.json", "w") as f:
            json.dump(progress, f, indent=2)


if __name__ == "__main__":
    config = {
        # Model
        "model_name": "smoke_test",
        "base_model": "Qwen/Qwen3-1.7B",
        
        # Dataset
        "hf_dataset": "TranDuong/medical-vlsp-2025",
        "dataset_dir": "dataset/medical-vlsp-2025",
        "num_parts": 10,
        
        # Training
        "num_gpus": 1,
        "output_dir": "outputs",
        
        # W&B
        "wandb_project": "qwen-mt",
        
        # Early stopping
        "patience": 2,
        "min_delta": 0.5,
    }
    
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
    
    orchestrator = TrainingOrchestrator(config)
    orchestrator.run_pipeline()