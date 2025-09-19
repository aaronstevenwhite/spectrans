#!/usr/bin/env python3
"""Basic FNet Example: Text Classification with Fourier Token Mixing.

This example demonstrates how to use FNet for a simple binary text classification task.
FNet replaces attention with 2D Fourier transforms, achieving O(n log n) complexity
while maintaining competitive performance.

Key concepts demonstrated:
- Creating an FNet model for classification
- Forward pass with token IDs and embeddings
- Basic training loop structure
- Model evaluation and inference

Requirements:
- spectrans
- torch
- numpy (optional, for data generation)
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from spectrans.models import FNet, FNetEncoder


def create_sample_data(
    vocab_size: int = 10000, num_samples: int = 1000, seq_len: int = 128, num_classes: int = 2
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic text classification data for demonstration.

    Args:
        vocab_size: Size of the vocabulary
        num_samples: Number of training samples
        seq_len: Sequence length
        num_classes: Number of classes

    Returns:
        Tuple of (input_ids, labels)
    """
    # Generate random token sequences
    input_ids = torch.randint(1, vocab_size, (num_samples, seq_len))

    # Create synthetic labels based on simple heuristics
    # (In practice, you'd have real labeled data)
    labels = torch.randint(0, num_classes, (num_samples,))

    return input_ids, labels


def basic_fnet_example():
    """Demonstrate basic FNet usage."""
    print("=== Basic FNet Example ===\n")

    # Model configuration
    vocab_size = 10000
    hidden_dim = 512
    num_layers = 6
    max_seq_length = 128
    num_classes = 2

    print("1. Creating FNet model...")
    model = FNet(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        max_sequence_length=max_seq_length,
        num_classes=num_classes,
        dropout=0.1,
        use_positional_encoding=True,
        positional_encoding_type="sinusoidal",
    )

    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("   Model complexity: O(n log n) where n = sequence length\n")

    # Sample data
    batch_size = 8
    seq_len = 64

    print("2. Forward pass with token IDs...")
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits = model(input_ids=input_ids)
        probs = torch.softmax(logits, dim=-1)

    print(f"   Input shape: {input_ids.shape}")
    print(f"   Output logits shape: {logits.shape}")
    print(f"   Sample probabilities: {probs[0].tolist()}\n")

    print("3. Forward pass with embeddings...")
    embeddings = torch.randn(batch_size, seq_len, hidden_dim)

    with torch.no_grad():
        logits = model(inputs_embeds=embeddings)

    print(f"   Embedding input shape: {embeddings.shape}")
    print(f"   Output shape: {logits.shape}\n")

    print("4. Using FNetEncoder (no classification head)...")
    encoder = FNetEncoder(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        max_sequence_length=max_seq_length,
        dropout=0.1,
    )

    with torch.no_grad():
        encoded = encoder(inputs_embeds=embeddings)

    print(f"   Encoder output shape: {encoded.shape}")
    print("   This gives you contextual embeddings for each token\n")


def training_example():
    """Demonstrate FNet training loop."""
    print("=== FNet Training Example ===\n")

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Model
    model = FNet(
        vocab_size=5000,
        hidden_dim=256,
        num_layers=4,
        max_sequence_length=64,
        num_classes=2,
        dropout=0.1,
    ).to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)

    # Sample training data
    train_data, train_labels = create_sample_data(
        vocab_size=5000, num_samples=100, seq_len=64, num_classes=2
    )
    train_data, train_labels = train_data.to(device), train_labels.to(device)

    print(f"Training data: {train_data.shape}")
    print(f"Training labels: {train_labels.shape}")

    # Training loop
    model.train()
    num_epochs = 5

    print(f"\nTraining for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        # Simple batching (in practice, use DataLoader)
        batch_size = 16
        for i in range(0, len(train_data), batch_size):
            batch_input = train_data[i : i + batch_size]
            batch_labels = train_labels[i : i + batch_size]

            # Forward pass
            logits = model(input_ids=batch_input)
            loss = F.cross_entropy(logits, batch_labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (recommended for transformer training)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"   Epoch {epoch + 1}: Average loss = {avg_loss:.4f}")

    print("\nTraining completed!\n")


def inference_example():
    """Demonstrate model inference and evaluation."""
    print("=== FNet Inference Example ===\n")

    # Create model
    model = FNet(
        vocab_size=5000,
        hidden_dim=256,
        num_layers=4,
        max_sequence_length=64,
        num_classes=3,  # Multi-class example
    )

    # Set to evaluation mode
    model.eval()

    # Sample test data
    test_input = torch.randint(1, 5000, (10, 32))  # 10 samples, length 32

    print("Performing inference...")
    with torch.no_grad():
        logits = model(input_ids=test_input)
        probs = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)

    print(f"Input shape: {test_input.shape}")
    print(f"Predictions: {predictions.tolist()}")
    print("Prediction probabilities (first 3 samples):")
    for i in range(3):
        print(f"   Sample {i + 1}: {probs[i].tolist()}")

    # Calculate accuracy (against random labels for demo)
    true_labels = torch.randint(0, 3, (10,))
    accuracy = (predictions == true_labels).float().mean()
    print(f"\nAccuracy on test set: {accuracy:.2%}")
    print("(Note: Random data, so accuracy will be around chance level)\n")


def model_comparison():
    """Compare FNet efficiency with standard transformer complexity."""
    print("=== Model Complexity Comparison ===\n")

    sequence_lengths = [128, 512, 1024, 2048]

    print("Theoretical complexity comparison:")
    print("FNet: O(n log n) - Fourier transforms")
    print("Standard Attention: O(n²) - Quadratic attention\n")

    print("Relative efficiency (FNet operations / Attention operations):")
    for n in sequence_lengths:
        fnet_ops = n * np.log2(n)
        attention_ops = n * n
        efficiency = fnet_ops / attention_ops
        print(f"   Sequence length {n:4d}: {efficiency:.4f} ({efficiency:.2%})")

    print("\nFNet becomes more efficient as sequence length increases!\n")


def main():
    """Run all examples."""
    print("FNet Examples with Spectrans\n")
    print("=" * 50)

    try:
        basic_fnet_example()
        training_example()
        inference_example()
        model_comparison()

        print("All examples completed successfully! ✓")
        print("\nNext steps:")
        print("- Try different model configurations")
        print("- Experiment with real datasets")
        print("- Compare with other spectral models (GFNet, AFNO)")
        print("- See config_usage.py for YAML-based configurations")

    except Exception as e:
        print(f"Error running example: {e}")
        print("Make sure spectrans is installed: pip install spectrans")
        raise


if __name__ == "__main__":
    main()
