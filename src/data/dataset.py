from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset
from ..core.base import BaseDataset

class InstructionDataset(BaseDataset):
    def __init__(self, data: List[Dict[str, str]], tokenizer, max_seq_len: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        # Simple instruction format: "Instruction: ... Input: ... Response: ..."
        text = f"Instruction: {item.get('instruction', '')}\nInput: {item.get('input', '')}\nResponse: {item.get('output', '')}"
        
        encodings = self.tokenizer.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_len,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        
        # Labels are same as input_ids for causal LM, but we might want to mask instruction
        labels = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
