#!/usr/bin/env python3
"""Integration Example: Complete Training Pipeline with Spectral Models.

This example demonstrates how to integrate Spectrans models into complete
machine learning pipelines, including:

- Data loading and preprocessing
- Training loops with validation
- Model checkpointing and resuming
- Mixed precision training
- Learning rate scheduling
- Model evaluation and metrics
- Deployment considerations

Key integration patterns:
- PyTorch Lightning integration
- Hugging Face Transformers compatibility
- Custom data loading for different tasks
- Production deployment patterns

Requirements:
- spectrans
- torch
- torchvision (for datasets)
- tqdm (for progress bars)
- sklearn (for metrics)
"""

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from spectrans.config import ConfigBuilder
from spectrans.models import AFNOModel, FNet, GFNet


@dataclass
class TrainingConfig:
    """Training configuration dataclass."""

    model_name: str = "fnet"
    vocab_size: int = 10000
    hidden_dim: int = 512
    num_layers: int = 8
    max_seq_length: int = 256
    num_classes: int = 2

    # Training hyperparameters
    batch_size: int = 32
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_steps: int = 500
    max_grad_norm: float = 1.0

    # System settings
    device: str = "auto"
    mixed_precision: bool = True
    gradient_checkpointing: bool = False
    dataloader_num_workers: int = 4

    # Checkpointing
    save_dir: str = "checkpoints"
    save_every_n_epochs: int = 2
    keep_n_checkpoints: int = 3


class SyntheticTextDataset(Dataset):
    """Synthetic text dataset for demonstration."""

    def __init__(
        self,
        vocab_size: int = 10000,
        seq_length: int = 256,
        num_samples: int = 10000,
        num_classes: int = 2,
        seed: int = 42,
    ):
        torch.manual_seed(seed)

        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.num_samples = num_samples
        self.num_classes = num_classes

        # Generate synthetic data with some structure
        self.data = []
        self.labels = []

        for i in range(num_samples):
            # Create sequences with class-dependent patterns
            if i % num_classes == 0:
                # Class 0: More low-frequency tokens
                tokens = torch.randint(1, vocab_size // 2, (seq_length,))
            else:
                # Other classes: More high-frequency tokens
                tokens = torch.randint(vocab_size // 2, vocab_size, (seq_length,))

            # Add some noise
            noise_mask = torch.rand(seq_length) < 0.1
            tokens[noise_mask] = torch.randint(1, vocab_size, (int(noise_mask.sum()),))

            self.data.append(tokens)
            self.labels.append(i % num_classes)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], torch.tensor(self.labels[idx], dtype=torch.long)


class ModelTrainer:
    """Complete training pipeline for spectral models."""

    def __init__(self, config: TrainingConfig):
        self.config = config

        # Setup device
        if config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device

        print(f"Using device: {self.device}")

        # Create model
        self.model = self._create_model()
        self.model.to(self.device)

        # Setup training components
        self.optimizer = None
        self.scheduler = None
        self.scaler = (
            GradScaler(device="cuda")
            if config.mixed_precision and torch.cuda.is_available()
            else None
        )

        # Tracking
        self.epoch = 0
        self.step = 0
        self.best_val_acc = 0.0
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}

        # Create save directory
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)

    def _create_model(self) -> nn.Module:
        """Create model based on configuration."""
        if self.config.model_name == "fnet":
            return FNet(
                vocab_size=self.config.vocab_size,
                hidden_dim=self.config.hidden_dim,
                num_layers=self.config.num_layers,
                max_sequence_length=self.config.max_seq_length,
                num_classes=self.config.num_classes,
                gradient_checkpointing=self.config.gradient_checkpointing,
            )
        elif self.config.model_name == "gfnet":
            return GFNet(
                vocab_size=self.config.vocab_size,
                hidden_dim=self.config.hidden_dim,
                num_layers=self.config.num_layers,
                max_sequence_length=self.config.max_seq_length,
                num_classes=self.config.num_classes,
            )
        elif self.config.model_name == "afno":
            return AFNOModel(
                vocab_size=self.config.vocab_size,
                hidden_dim=self.config.hidden_dim,
                num_layers=self.config.num_layers,
                max_sequence_length=self.config.max_seq_length,
                num_classes=self.config.num_classes,
                modes_seq=min(32, self.config.max_seq_length // 8),
            )
        else:
            raise ValueError(f"Unknown model: {self.config.model_name}")

    def setup_optimization(self, total_steps: int):
        """Setup optimizer and scheduler."""
        # Optimizer with weight decay
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Learning rate scheduler
        pct_start = min(0.3, self.config.warmup_steps / total_steps)  # Cap at 30%
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy="cos",
        )

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch}")

        for _batch_idx, (input_ids, labels) in enumerate(pbar):
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)

            if self.optimizer is None:
                raise RuntimeError("Optimizer not initialized. Call setup_optimization first.")

            self.optimizer.zero_grad()

            # Forward pass with optional mixed precision
            if self.scaler is not None:
                with autocast(device_type="cuda"):
                    logits = self.model(input_ids=input_ids)
                    loss = F.cross_entropy(logits, labels)

                # Backward pass
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(input_ids=input_ids)
                loss = F.cross_entropy(logits, labels)

                loss.backward()

                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )

                self.optimizer.step()

            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.step += 1

            # Update progress bar
            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"}
            )

        return total_loss / num_batches

    def validate(self, val_loader: DataLoader) -> tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for input_ids, labels in tqdm(val_loader, desc="Validating"):
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                if self.scaler is not None:
                    with autocast(device_type="cuda"):
                        logits = self.model(input_ids=input_ids)
                        loss = F.cross_entropy(logits, labels)
                else:
                    logits = self.model(input_ids=input_ids)
                    loss = F.cross_entropy(logits, labels)

                total_loss += loss.item()

                # Calculate accuracy
                pred = logits.argmax(dim=-1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def save_checkpoint(self, filepath: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.epoch,
            "step": self.step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "best_val_acc": self.best_val_acc,
            "config": self.config,
            "history": self.history,
        }

        torch.save(checkpoint, filepath)

        if is_best:
            best_path = Path(filepath).parent / "best_model.pt"
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is not None and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.scaler and checkpoint["scaler_state_dict"]:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.epoch = checkpoint["epoch"]
        self.step = checkpoint["step"]
        self.best_val_acc = checkpoint["best_val_acc"]
        self.history = checkpoint["history"]

    def train(
        self, train_loader: DataLoader, val_loader: DataLoader, resume_from: str | None = None
    ):
        """Complete training loop."""
        # Resume from checkpoint if specified
        if resume_from and Path(resume_from).exists():
            print(f"Resuming from checkpoint: {resume_from}")
            self.load_checkpoint(resume_from)

        # Setup optimization
        total_steps = len(train_loader) * self.config.num_epochs
        if self.optimizer is None:
            self.setup_optimization(total_steps)

        print(f"Training {self.config.model_name} for {self.config.num_epochs} epochs")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch

            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_loss, val_acc = self.validate(val_loader)

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            # Print epoch results
            print(
                f"Epoch {epoch}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"val_acc={val_acc:.4f}"
            )

            # Save checkpoint
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc

            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                checkpoint_path = Path(self.config.save_dir) / f"checkpoint_epoch_{epoch}.pt"
                self.save_checkpoint(str(checkpoint_path), is_best)

        print(f"Training completed! Best validation accuracy: {self.best_val_acc:.4f}")


def demonstrate_basic_training():
    """Basic training example."""
    print("=== Basic Training Pipeline ===\n")

    # Configuration
    config = TrainingConfig(
        model_name="fnet", batch_size=16, num_epochs=3, learning_rate=1e-4, max_seq_length=128
    )

    # Create synthetic dataset
    full_dataset = SyntheticTextDataset(
        vocab_size=config.vocab_size,
        seq_length=config.max_seq_length,
        num_samples=1000,
        num_classes=config.num_classes,
    )

    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues in example
    )
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    print(f"Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")

    # Train model
    trainer = ModelTrainer(config)
    trainer.train(train_loader, val_loader)

    print("✓ Basic training completed\n")


def demonstrate_model_comparison():
    """Compare different spectral models."""
    print("=== Model Comparison ===\n")

    models = ["fnet", "gfnet", "afno"]
    results = {}

    # Create small dataset for quick comparison
    dataset = SyntheticTextDataset(num_samples=200, seq_length=64)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    for model_name in models:
        print(f"Training {model_name.upper()}...")

        config = TrainingConfig(
            model_name=model_name,
            batch_size=8,
            num_epochs=2,
            learning_rate=2e-4,
            max_seq_length=64,
            hidden_dim=256,
            num_layers=4,
        )

        trainer = ModelTrainer(config)

        # Train briefly
        trainer.train(train_loader, val_loader)

        # Record results
        results[model_name] = {
            "params": sum(p.numel() for p in trainer.model.parameters()),
            "best_acc": trainer.best_val_acc,
            "final_train_loss": trainer.history["train_loss"][-1],
            "final_val_loss": trainer.history["val_loss"][-1],
        }

    # Print comparison
    print("\nModel Comparison Results:")
    print(f"{'Model':<10} {'Params':<10} {'Val Acc':<10} {'Train Loss':<12} {'Val Loss':<10}")
    print("-" * 60)

    for model_name, metrics in results.items():
        print(
            f"{model_name.upper():<10} "
            f"{metrics['params']:>8,} "
            f"{metrics['best_acc']:<10.4f} "
            f"{metrics['final_train_loss']:<12.4f} "
            f"{metrics['final_val_loss']:<10.4f}"
        )

    print("\n✓ Model comparison completed\n")


def demonstrate_config_based_training():
    """Training using YAML configuration."""
    print("=== Configuration-Based Training ===\n")

    try:
        # Load model from config
        builder = ConfigBuilder()
        model = builder.build_model("configs/fnet.yaml")

        print("✓ Model loaded from YAML configuration")
        print(f"✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # You could integrate this model into the training pipeline
        # by modifying ModelTrainer to accept pre-built models

    except FileNotFoundError:
        print("⚠ No config file found, creating example inline config...")

        # Create and use inline configuration
        from spectrans.config.models import FNetModelConfig

        config = FNetModelConfig(
            hidden_dim=384, num_layers=6, sequence_length=256, vocab_size=15000, num_classes=3
        )

        from spectrans.config import build_model_from_config

        model = build_model_from_config({"model": config.model_dump()})

        print("✓ Model created from programmatic configuration")
        print(f"✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("✓ Configuration-based training demo completed\n")


def demonstrate_production_patterns():
    """Show production deployment patterns."""
    print("=== Production Patterns ===\n")

    # Model export/import
    print("1. Model serialization...")

    model = FNet(
        vocab_size=5000, hidden_dim=256, num_layers=4, max_sequence_length=128, num_classes=2
    )

    # Save for production
    save_path = "production_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "vocab_size": 5000,
                "hidden_dim": 256,
                "num_layers": 4,
                "max_sequence_length": 128,
                "num_classes": 2,
            },
        },
        save_path,
    )

    print(f"   ✓ Model saved to {save_path}")

    # Load in production
    checkpoint = torch.load(save_path, map_location="cpu")
    production_model = FNet(**checkpoint["model_config"])
    production_model.load_state_dict(checkpoint["model_state_dict"])
    production_model.eval()

    print("   ✓ Model loaded for production")

    # Inference optimization
    print("\n2. Inference optimization...")

    # JIT compilation for faster inference
    sample_input = torch.randint(1, 5000, (1, 64))

    with torch.no_grad():
        traced_model = torch.jit.trace(production_model, sample_input)
        traced_output = traced_model(sample_input)

    print("   ✓ Model traced with TorchScript")
    print(f"   ✓ Output shape: {traced_output.shape}")

    # Batch inference
    print("\n3. Batch inference...")

    batch_inputs = torch.randint(1, 5000, (8, 64))
    with torch.no_grad():
        batch_outputs = production_model(input_ids=batch_inputs)

    print(f"   ✓ Batch processed: {batch_inputs.shape} -> {batch_outputs.shape}")

    # Clean up
    Path(save_path).unlink()

    print("\n✓ Production patterns demo completed\n")


def main():
    """Run all integration examples."""
    print("Spectral Models Integration Examples\n")
    print("=" * 50)

    try:
        demonstrate_basic_training()
        demonstrate_model_comparison()
        demonstrate_config_based_training()
        demonstrate_production_patterns()

        print("All integration examples completed successfully! ✓")

        print("\nKey integration points:")
        print("- Standard PyTorch training loops work seamlessly")
        print("- Mixed precision training supported")
        print("- Models integrate with existing ML pipelines")
        print("- Production deployment follows standard patterns")

        print("\nNext steps for production use:")
        print("- Implement proper data loading for your domain")
        print("- Add comprehensive evaluation metrics")
        print("- Set up experiment tracking (wandb, tensorboard)")
        print("- Consider distributed training for large models")

    except Exception as e:
        print(f"Error running integration examples: {e}")
        print("Make sure all dependencies are installed")
        raise


if __name__ == "__main__":
    main()
