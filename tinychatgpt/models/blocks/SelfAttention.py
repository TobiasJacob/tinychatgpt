import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionBlock(nn.Module):
    def __init__(self, n_embed: int, head_size: int, max_seq_length: int) -> None:
        super().__init__()
        self.key = nn.Linear(n_embed, head_size)
        self.query = nn.Linear(n_embed, head_size)
        self.value = nn.Linear(n_embed, head_size)
        self.register_buffer("mask", torch.triu(torch.ones(max_seq_length, max_seq_length).bool(), diagonal=1))
        
    def forward(self, x, mask=None) -> None:
        # x.shape = (batch_size, seq_len, n_embed)
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        # k.shape = q.shape = v.shape = (batch_size, seq_len, head_size)
        weights = torch.einsum("bti,bui->btu", q, k)
        # weights.shape = (batch_size, seq_len, seq_len)
        weights.masked_fill_(self.mask[None, :T, :T], float("-inf"))
        weights = F.softmax(weights, dim=2)
        # weights.shape = (batch_size, seq_len, seq_len)
        
        out = torch.einsum("btu,btj->btj", weights, v)
        # out.shape = (batch_size, seq_len, head_size)
        return out
        
        
