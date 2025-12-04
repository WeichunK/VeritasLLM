import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        # Custom hidden dim logic can be added here
        
        self.w1 = nn.Linear(config.dim, hidden_dim, bias=config.use_bias)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=config.use_bias)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=config.use_bias)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.moe.num_experts
        self.num_experts_per_tok = config.moe.num_experts_per_tok
        self.experts = nn.ModuleList([FeedForward(config) for _ in range(self.num_experts)])
        self.gate = nn.Linear(config.dim, self.num_experts, bias=False)

    def forward(self, x):
        bsz, seqlen, dim = x.shape
        x_flat = x.view(-1, dim)
        
        gate_logits = self.gate(x_flat)
        routing_weights, selected_experts = torch.topk(gate_logits, self.num_experts_per_tok, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float).to(x.dtype)
        
        results = torch.zeros_like(x_flat)
        
        # Naive loop implementation (can be optimized with scatter/gather)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            if batch_idx.numel() > 0:
                results[batch_idx] += routing_weights[batch_idx, nth_expert, None] * expert(x_flat[batch_idx])
                
        return results.view(bsz, seqlen, dim)
