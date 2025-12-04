from typing import List, Union
import torch
from transformers import AutoTokenizer
from ..core.base import BaseTokenizer

class TokenizerWrapper(BaseTokenizer):
    def __init__(self, model_name_or_path: str, use_fast: bool = True):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode(self, text: Union[str, List[str]], **kwargs) -> torch.Tensor:
        return self.tokenizer(text, return_tensors="pt", **kwargs)["input_ids"]

    def decode(self, token_ids: torch.Tensor, **kwargs) -> str:
        return self.tokenizer.decode(token_ids, **kwargs)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size
    
    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id
    
    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id
