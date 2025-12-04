# VeritasLLM Usage Guide

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repo_url>
    cd VeritasLLM
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Quick Start

To start training with the default configuration:

```bash
python3 scripts/train.py
```

## Configuration

VeritasLLM uses **Hydra** for configuration. You can override any parameter from the command line.

### Common Overrides

*   **Change Model Size**:
    ```bash
    python3 scripts/train.py model.n_layers=12 model.dim=768
    ```

*   **Enable FlashAttention**:
    ```bash
    python3 scripts/train.py model.attention_backend=flash_attention
    ```

*   **Enable Mixture of Experts (MoE)**:
    ```bash
    python3 scripts/train.py model.moe.enabled=true model.moe.num_experts=8
    ```

*   **Change Precision**:
    ```bash
    python3 scripts/train.py training.precision=fp16
    ```

## Training Control

VeritasLLM supports graceful stopping and resuming.

*   **Graceful Stop**: Press `Ctrl+C` **once**. The trainer will finish the current batch, save a checkpoint (`checkpoint_stopped.pt`), and exit.
*   **Stop via File**: Create a file named `stop.signal` in the root directory.
*   **Resume**:
    ```bash
    python3 scripts/train.py training.resume_from_checkpoint=checkpoint_stopped.pt
    ```

## Monitoring

Metrics are automatically logged to **WandB**.
*   **Loss Curves**: Real-time training and validation loss.
*   **System Metrics**: GPU usage, memory, etc.
*   **Git Info**: Every run is tagged with the Git commit hash for reproducibility.

## Advanced Features

### Internals Observation

You can inspect internal model states using the `InternalsObserver`.

```python
from src.model import observer

# Register a hook on a specific layer
def my_hook(module, input, output):
    print(f"Layer output mean: {output.mean()}")

model.layers[0].register_forward_hook(my_hook)
```

### Adding New Datasets

Inherit from `BaseDataset` in `src/core/base.py` and implement `__len__` and `__getitem__`.

```python
from src.core.base import BaseDataset

class MyDataset(BaseDataset):
    ...
```
