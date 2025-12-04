import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import os
from tqdm import tqdm
from ..core.base import BaseTrainer
from .controller import TrainingController
from ..utils.experiment import ExperimentManager

logger = logging.getLogger(__name__)

class Trainer(BaseTrainer):
    def __init__(self, model, train_dataset, config, experiment_manager: ExperimentManager):
        self.model = model
        self.config = config
        self.experiment_manager = experiment_manager
        self.controller = TrainingController()
        
        self.device = torch.device(config.training.device)
        self.model.to(self.device)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        self.global_step = 0

    def train(self):
        self.model.train()
        logger.info("Starting training...")
        
        progress_bar = tqdm(total=self.config.training.max_steps)
        
        while self.global_step < self.config.training.max_steps:
            for batch in self.train_loader:
                if self.controller.should_stop():
                    logger.info("Stopping training early...")
                    self.save_checkpoint("checkpoint_stopped.pt")
                    return

                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids)
                logits = outputs["logits"]
                
                # Calculate loss
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )
                
                # Backward pass
                loss.backward()
                
                if (self.global_step + 1) % self.config.training.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                # Logging
                if self.global_step % self.config.training.log_interval == 0:
                    self.experiment_manager.log_metrics({"train/loss": loss.item()}, step=self.global_step)
                    progress_bar.set_description(f"Loss: {loss.item():.4f}")
                
                # Checkpointing
                if self.global_step % self.config.training.save_interval == 0:
                    self.save_checkpoint(f"checkpoint_{self.global_step}.pt")
                
                self.global_step += 1
                progress_bar.update(1)
                
                if self.global_step >= self.config.training.max_steps:
                    break
        
        self.save_checkpoint("checkpoint_final.pt")
        logger.info("Training finished.")

    def save_checkpoint(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'global_step': self.global_step
        }, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        logger.info(f"Loaded checkpoint from {path}")
