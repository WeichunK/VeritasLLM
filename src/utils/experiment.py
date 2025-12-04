import os
import git
import logging
import wandb
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

class ExperimentManager:
    """
    Manages experiment tracking, including Git commit hash, config logging, and WandB integration.
    """
    def __init__(self, config: DictConfig):
        self.config = config
        self.project_name = config.project_name
        self.run_dir = os.getcwd()
        
        # Initialize WandB
        if config.get("debug", False):
            os.environ["WANDB_MODE"] = "offline"
            
        wandb.init(
            project=self.project_name,
            config=OmegaConf.to_container(config, resolve=True),
            dir=self.run_dir
        )
        
        self._log_git_info()

    def _log_git_info(self):
        try:
            repo = git.Repo(search_parent_directories=True)
            sha = repo.head.object.hexsha
            wandb.config.update({"git_commit": sha})
            logger.info(f"Git commit: {sha}")
        except Exception as e:
            logger.warning(f"Failed to get git info: {e}")

    def log_metrics(self, metrics: dict, step: int):
        wandb.log(metrics, step=step)

    def finish(self):
        wandb.finish()
