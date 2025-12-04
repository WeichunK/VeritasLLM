# VeritasLLM Architecture

## Overview

## System Architecture

```mermaid
graph TB
    subgraph Configuration
        Config[Hydra Config]
        CLI[Command Line Overrides]
    end

    subgraph Data_Pipeline
        RawData[Raw Text/JSON]
        Tokenizer[TokenizerWrapper]
        Dataset[InstructionDataset]
        Loader[DataLoader]
    end

    subgraph Model_Architecture
        Embed[Embeddings]
        Block[TransformerBlock]
        Attn[ConfigurableAttention]
        FFN[FeedForward / MoE]
        Head[Output Head]
        Observer[InternalsObserver]
    end

    subgraph Training_Loop
        Trainer[Trainer]
        Optim[Optimizer (AdamW)]
        Loss[Loss Function]
        Controller[TrainingController]
    end

    subgraph Experiment_Tracking
        WandB[WandB Logger]
        Git[Git Tracker]
    end

    CLI --> Config
    Config --> Trainer
    Config --> Model_Architecture
    Config --> Data_Pipeline

    RawData --> Tokenizer
    Tokenizer --> Dataset
    Dataset --> Loader
    Loader --> Trainer

    Trainer -->|Forward Pass| Embed
    Embed --> Block
    Block --> Attn
    Block --> FFN
    FFN --> Block
    Block --> Head
    Head --> Loss
    Loss -->|Backward Pass| Optim
    Optim -->|Update Weights| Model_Architecture

    Controller -->|Stop Signal| Trainer
    Observer -.->|Inspect| Block
    Observer -.->|Inspect| Attn

    Trainer -->|Log Metrics| WandB
    Trainer -->|Log Config| Git
```

## Class Diagram

## Key Modules

### 1. Configuration (`config/`)
*   **Hydra** based.
*   Hierarchical structure: `model`, `data`, `training`, `experiment`.
*   Supports variable interpolation and sweeps.

### 2. Core (`src/core/`)
*   Defines abstract base classes (`BaseModel`, `BaseTrainer`).
*   Ensures consistent interfaces across different implementations.

### 3. Model (`src/model/`)
*   **`ConfigurableAttention`**: Supports `flash_attention`, `sdpa` (PyTorch 2.0), and vanilla math.
*   **`MoELayer`**: Implements sparse Mixture of Experts with top-k routing.
*   **`InternalsObserver`**: A singleton hook manager to probe any part of the model without changing code.

### 4. Training (`src/training/`)
*   **`Trainer`**: Handles the training loop, gradient accumulation, and mixed precision (BF16/FP16).
*   **`TrainingController`**: Manages signals (SIGINT) for safe termination.

### 5. Utils (`src/utils/`)
*   **`ExperimentManager`**: Integrates with WandB and Git to track every run's metadata.
