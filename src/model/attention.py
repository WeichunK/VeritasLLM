import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class ConfigurableAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads
        self.n_kv_heads = config.n_kv_heads if config.n_kv_heads is not None else config.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        
        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=config.use_bias)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=config.use_bias)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=config.use_bias)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=config.use_bias)
        
        self.backend = config.attention_backend
        self.dropout = config.dropout

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # Apply RoPE (simplified placeholder)
        if freqs_cis is not None:
            # Apply rotary embeddings here
            pass

        # KV Cache handling would go here for inference

        # Repeat KV heads if GQA
        if self.n_rep > 1:
            xk = xk.repeat_interleave(self.n_rep, dim=2)
            xv = xv.repeat_interleave(self.n_rep, dim=2)

        xq = xq.transpose(1, 2)  # (bsz, n_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        if self.backend == "flash_attention":
             # Use PyTorch 2.0 SDPA which supports FlashAttention automatically
            output = F.scaled_dot_product_attention(
                xq, xk, xv, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
            )
        else:
            # Manual implementation
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = F.dropout(scores, p=self.dropout, training=self.training)
            output = torch.matmul(scores, xv)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)
