import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import torch
from src.core.config import log_config
from src.model import VeritasModel
from src.data import TokenizerWrapper, InstructionDataset
from src.training import Trainer
from src.utils.experiment import ExperimentManager

logger = logging.getLogger(__name__)

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    log_config(cfg)
    
    # Initialize Experiment Manager
    experiment_manager = ExperimentManager(cfg)
    
    # Initialize Tokenizer
    # For demo purposes, we use a dummy tokenizer or a real one if path provided
    # Here we assume a default small model for testing if not specified
    tokenizer_path = cfg.data.get("tokenizer_path", "gpt2")
    tokenizer = TokenizerWrapper(tokenizer_path)
    
    # Initialize Dataset
    # Dummy data for demonstration
    dummy_data = [
        {"instruction": "Hello", "input": "", "output": "World"}
    ] * 100
    dataset = InstructionDataset(dummy_data, tokenizer, cfg.model.max_seq_len)
    
    # Initialize Model
    logger.info(f"Initializing model: {cfg.model.name}")
    model = VeritasModel(cfg.model)
    
    # Initialize Trainer
    trainer = Trainer(model, dataset, cfg, experiment_manager)
    
    # Start Training
    trainer.train()
    
    experiment_manager.finish()

if __name__ == "__main__":
    main()
