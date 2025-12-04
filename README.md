# VeritasLLM

**VeritasLLM** is a comprehensive, modular, and highly configurable framework for Large Language Model (LLM) development. It supports the entire lifecycle of an LLM, including:

*   **Data Processing**: Raw data ingestion, cleaning, and tokenization.
*   **Training**: Pre-training and fine-tuning with advanced efficiency techniques (FlashAttention, Mixed Precision).
*   **Alignment**: RLHF (PPO) and DPO.
*   **Evaluation**: Integration with standard benchmarks (via `lm-evaluation-harness`) and custom metrics.
*   **Observability**: Deep inspection of internal model states (attention maps, expert routing).
*   **Control**: Real-time monitoring and manual intervention capabilities.

## Architecture

VeritasLLM is built on **PyTorch** and uses **Hydra** for flexible configuration.

```mermaid
graph TD
    Config[Hydra Config] --> Trainer
    Data[Data Module] --> Trainer
    Model[Model Architecture] --> Trainer
    Trainer --> |Updates| Model
    Trainer --> |Logs| ExperimentManager[Experiment Manager (WandB/MLflow)]
    Trainer --> |Checkpoints| Storage
    Controller[Training Controller] --> |Signals| Trainer
    Observer[Internals Observer] -.-> |Probes| Model
```

## Features

*   **Modular Design**: Easily swap out attention mechanisms, loss functions, or entire model blocks.
*   **Efficiency**: Support for FlashAttention, SDPA, and quantization (4-bit/8-bit).
*   **Advanced Architectures**: Native support for Mixture of Experts (MoE) and Multimodal extensions.
*   **Reproducibility**: Automatic tracking of configurations, data hashes, and git commits.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Training

```bash
python scripts/train.py experiment=sample_experiment
```
