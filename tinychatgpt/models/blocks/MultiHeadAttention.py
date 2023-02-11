import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, head_size: int, max_seq_length: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_size = head_size
        self.key = nn.Linear(head_size, head_size)
        self.query = nn.Linear(head_size, head_size)
        self.value = nn.Linear(head_size, head_size)
        self.proj = nn.Linear(head_size, head_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, x1, x2, attn_mask) -> None:
        # x.shape = (batch_size, seq_len, head_size)
        # attn_mask.shape = (batch_size, seq_len, seq_len)
        B,T,C = x.shape
        single_head_size = self.head_size // self.n_heads
        k = self.key(x).reshape(B, T, self.n_heads, single_head_size)
        q = self.query(x).reshape(B, T, self.n_heads, single_head_size)
        v = self.value(x).reshape(B, T, self.n_heads, single_head_size)
        # k.shape = q.shape = v.shape = (batch_size, seq_len, n_heads, single_head_size)
        weights = torch.einsum("bkhi,bqhi->bkqh", k, q)
        # weights.shape = (batch_size, seq_len, seq_len, n_heads)
        weights.masked_fill_(attn_mask[None, :T, :T, None], float("-inf"))
        weights = weights / (self.head_size ** 0.5)
        weights = F.softmax(weights, dim=2) # q sums to 1
        # weights.shape = (batch_size, seq_len, seq_len, n_heads)
        
        out = torch.einsum("bkqh,bqhj->bkhj", weights, v)
        # out.shape = (batch_size, seq_len, n_heads, single_head_size)
        out = out.reshape(B, T, self.head_size)
        # out.shape = (batch_size, seq_len, head_size)
        out = self.proj(out)
        out = self.dropout(out)
        return (out, weights)

