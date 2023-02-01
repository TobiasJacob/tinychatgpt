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
        self.register_buffer("mask", torch.triu(torch.ones(max_seq_length, max_seq_length).bool(), diagonal=1))
        self.proj = nn.Linear(head_size, head_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None) -> None:
        # x.shape = (batch_size, seq_len, head_size)
        B,T,C = x.shape
        k = self.key(x).reshape(B, T, self.n_heads, self.head_size // self.n_heads)
        q = self.query(x).reshape(B, T, self.n_heads, self.head_size // self.n_heads)
        v = self.value(x).reshape(B, T, self.n_heads, self.head_size // self.n_heads)
        # k.shape = q.shape = v.shape = (batch_size, seq_len, n_heads, single_head_size)
        weights = torch.einsum("bthi,buhi->btuh", q, k)
        # weights.shape = (batch_size, seq_len, seq_len, n_heads)
        weights.masked_fill_(self.mask[None, :T, :T, None], float("-inf"))
        weights = F.softmax(weights, dim=2)
        # weights.shape = (batch_size, seq_len, seq_len, n_heads)
        
        out = torch.einsum("btuh,bthj->btjh", weights, v)
        # out.shape = (batch_size, seq_len, n_heads, single_head_size)
        out = out.reshape(B, T, self.head_size)
        # out.shape = (batch_size, seq_len, head_size)
        out = self.proj(out)
        out = self.dropout(out)
        return out

