from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import torch
from torch.utils.data import Dataset

class BaseModel(torch.nn.Module, ABC):
    """Abstract base class for all models in VeritasLLM."""
    
    @abstractmethod
    def forward(self, input_ids: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        Returns a dictionary containing 'logits', 'loss' (optional), and other outputs.
        """
        pass

    @abstractmethod
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, **kwargs) -> torch.Tensor:
        """
        Generate text from the model.
        """
        pass

class BaseTokenizer(ABC):
    """Abstract base class for tokenizers."""
    
    @abstractmethod
    def encode(self, text: str, **kwargs) -> torch.Tensor:
        pass
    
    @abstractmethod
    def decode(self, token_ids: torch.Tensor, **kwargs) -> str:
        pass
    
    @property
    @abstractmethod
    def vocab_size(self) -> int:
        pass

class BaseDataset(Dataset, ABC):
    """Abstract base class for datasets."""
    pass

class BaseTrainer(ABC):
    """Abstract base class for trainers."""
    
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def save_checkpoint(self, path: str):
        pass
    
    @abstractmethod
    def load_checkpoint(self, path: str):
        pass
