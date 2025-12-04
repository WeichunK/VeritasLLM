import torch
import torch.nn as nn
from ..core.base import BaseModel
from .attention import ConfigurableAttention
from .layers import FeedForward, MoELayer

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        
        self.attention = ConfigurableAttention(config)
        
        if config.moe.enabled:
            self.feed_forward = MoELayer(config)
        else:
            self.feed_forward = FeedForward(config)
            
        self.attention_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: torch.Tensor):
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class VeritasModel(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers
        
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([TransformerBlock(i, config) for i in range(config.n_layers)])
        self.norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        self.freqs_cis = None # Precompute RoPE here

    def forward(self, input_ids: torch.Tensor, **kwargs):
        bsz, seqlen = input_ids.shape
        h = self.tok_embeddings(input_ids)
        
        # Create mask (simplified)
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=input_ids.device)
            mask = torch.triu(mask, diagonal=1)
            
        # RoPE freqs would be computed/retrieved here
        freqs_cis = None 

        for layer in self.layers:
            h = layer(h, freqs_cis, mask)
            
        h = self.norm(h)
        logits = self.output(h)
        
        return {"logits": logits}

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, **kwargs) -> torch.Tensor:
        # Simple greedy generation for now
        for _ in range(max_new_tokens):
            logits = self(input_ids)["logits"]
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids
