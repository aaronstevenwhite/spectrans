# Quick Start Guide

This tutorial will get you up and running with Spectrans in minutes. We'll cover the essential patterns for using spectral transformers in your projects.

## Prerequisites

Make sure you have Spectrans installed. If not, see the [Installation Guide](installation.md).

```bash
pip install spectrans
```

## Your First Spectral Model

Let's start with the most straightforward approach - creating an FNet model for text classification:

```python
import torch
from spectrans.models import FNet

# Create an FNet model for binary classification
model = FNet(
    vocab_size=30000,        # Size of your vocabulary
    hidden_dim=768,          # Hidden dimension (like BERT-base)
    num_layers=12,           # Number of transformer layers
    max_sequence_length=512, # Maximum input length
    num_classes=2            # Binary classification
)

# Create some sample input
batch_size = 4
seq_len = 128
input_ids = torch.randint(0, 30000, (batch_size, seq_len))

# Forward pass
logits = model(input_ids=input_ids)
print(f"Output shape: {logits.shape}")  # (4, 2)

# Apply softmax for probabilities
probs = torch.softmax(logits, dim=-1)
print(f"Probabilities: {probs}")
```

## Using Pre-Embedded Inputs

If you already have embeddings (from a pre-trained model, for example):

```python
from spectrans.models import FNetEncoder

# For encoder-only usage (no classification head)
encoder = FNetEncoder(
    hidden_dim=768,
    num_layers=12,
    max_sequence_length=512,
    dropout=0.1
)

# Use with embeddings directly
embeddings = torch.randn(4, 128, 768)  # (batch, seq_len, hidden_dim)
encoded = encoder(inputs_embeds=embeddings)
print(f"Encoded shape: {encoded.shape}")  # (4, 128, 768)
```

## Configuration-Based Models

Spectrans provides YAML-based configuration for easy experimentation:

```python
from spectrans.config import ConfigBuilder

# Load a pre-defined configuration
builder = ConfigBuilder()
model = builder.build_model("examples/configs/fnet.yaml")

# The config file defines all model parameters
# You can also create custom configs
```

Here's what a configuration file looks like:

```yaml
# examples/configs/my_fnet.yaml
model:
  model_type: "fnet"
  hidden_dim: 512
  num_layers: 8
  sequence_length: 256
  dropout: 0.1
  vocab_size: 20000
  num_classes: 5
  ffn_hidden_dim: 2048
  use_real_fft: true

layers:
  mixing:
    type: "fourier"
    hidden_dim: 512
    dropout: 0.1
    fft_norm: "ortho"
```

## Exploring Different Architectures

### GFNet (Global Filter Networks)

```python
from spectrans.models import GFNet

# GFNet uses learnable filters in frequency domain
gfnet = GFNet(
    vocab_size=30000,
    hidden_dim=768,
    num_layers=12,
    max_sequence_length=512,
    num_classes=10  # Multi-class classification
)

input_ids = torch.randint(0, 30000, (2, 256))
logits = gfnet(input_ids=input_ids)
print(f"GFNet output: {logits.shape}")  # (2, 10)
```

### AFNO (Adaptive Fourier Neural Operator)

```python
from spectrans.models import AFNOModel

# AFNO is great for tasks requiring global context
afno = AFNOModel(
    vocab_size=30000,
    hidden_dim=768,
    num_layers=8,
    max_sequence_length=1024,  # Longer sequences
    num_classes=3,
    n_modes=64  # Number of Fourier modes to retain
)

# AFNO excels with longer sequences
input_ids = torch.randint(0, 30000, (1, 1024))
logits = afno(input_ids=input_ids)
print(f"AFNO output: {logits.shape}")  # (1, 3)
```

### Wavelet Transformers

```python
from spectrans.models import WaveletTransformer

# Wavelet transforms provide multi-resolution analysis
wavelet_model = WaveletTransformer(
    vocab_size=30000,
    hidden_dim=512,
    num_layers=10,
    max_sequence_length=512,
    num_classes=4,
    wavelet_type="db4",  # Daubechies-4 wavelet
    decomposition_levels=3
)

input_ids = torch.randint(0, 30000, (2, 256))
logits = wavelet_model(input_ids=input_ids)
print(f"Wavelet output: {logits.shape}")  # (2, 4)
```

## Working with Individual Components

For more control, you can compose models from individual components:

```python
from spectrans.layers.mixing import FourierMixing, WaveletMixing
from spectrans.blocks import SpectralTransformerBlock
from spectrans.models.base import BaseModel

# Create custom mixing layers
fourier_mixing = FourierMixing(hidden_dim=768, dropout=0.1)
wavelet_mixing = WaveletMixing(hidden_dim=768, wavelet="db6", levels=4)

# Create transformer blocks
fourier_block = SpectralTransformerBlock(
    mixing_layer=fourier_mixing,
    hidden_dim=768,
    ffn_hidden_dim=3072
)

wavelet_block = SpectralTransformerBlock(
    mixing_layer=wavelet_mixing,
    hidden_dim=768,
    ffn_hidden_dim=3072
)

# Combine in a custom model (you would need to implement the full model class)
```

## Training Your Model

Here's a simple training loop example:

```python
import torch.nn.functional as F
from torch.optim import AdamW

# Create model and data
model = FNet(vocab_size=10000, hidden_dim=512, num_layers=6, 
             max_sequence_length=256, num_classes=2)
optimizer = AdamW(model.parameters(), lr=1e-4)

# Sample training step
def training_step(model, batch):
    input_ids, labels = batch
    
    # Forward pass
    logits = model(input_ids=input_ids)
    
    # Compute loss
    loss = F.cross_entropy(logits, labels)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

# Example batch
batch_size = 8
input_ids = torch.randint(0, 10000, (batch_size, 128))
labels = torch.randint(0, 2, (batch_size,))

loss = training_step(model, (input_ids, labels))
print(f"Training loss: {loss:.4f}")
```

## GPU Usage

All models support GPU acceleration automatically:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = FNet(vocab_size=30000, hidden_dim=768, num_layers=12, 
             max_sequence_length=512, num_classes=2)
model = model.to(device)

# Data will need to be moved to GPU as well
input_ids = torch.randint(0, 30000, (4, 128)).to(device)
logits = model(input_ids=input_ids)
```

## Memory Optimization

For large models or long sequences:

```python
# Enable gradient checkpointing to save memory
model = FNet(vocab_size=30000, hidden_dim=1024, num_layers=24,
             max_sequence_length=1024, num_classes=2,
             gradient_checkpointing=True)

# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    logits = model(input_ids=input_ids)
    loss = F.cross_entropy(logits, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## What's Next?

Now that you've got the basics:

1. **Explore Examples**: Check out the [examples](../examples/) directory for complete applications
2. **Read the API Documentation**: Detailed reference at [API docs](api/index.md)  
3. **Experiment with Configurations**: Modify the YAML configs in `examples/configs/` to try different architectures
4. **Build Custom Components**: Learn how to create your own spectral layers

## Key Concepts to Remember

- **Direct Instantiation**: Use model classes like `FNet()`, `GFNet()` directly
- **Configuration**: Use YAML files with `ConfigBuilder` for experiments
- **Components**: Mix and match layers, blocks, and transforms
- **Efficiency**: Spectral methods offer :math:`O(n \log n)` complexity vs :math:`O(n^2)` for attention
- **Flexibility**: Works with both token IDs and pre-computed embeddings

Happy building with Spectrans! 🚀