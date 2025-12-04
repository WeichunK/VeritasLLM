from omegaconf import DictConfig, OmegaConf
import logging

logger = logging.getLogger(__name__)

def log_config(cfg: DictConfig):
    """Logs the current configuration."""
    logger.info("Configuration:\n" + OmegaConf.to_yaml(cfg))

def save_config(cfg: DictConfig, path: str):
    """Saves the configuration to a file."""
    OmegaConf.save(cfg, path)
