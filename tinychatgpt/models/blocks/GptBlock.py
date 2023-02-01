import torch
import torch.nn as nn
import torch.nn.functional as F
from tinychatgpt.models.blocks.FeedForward import FeedForward

from tinychatgpt.models.blocks.MultiHeadAttention import MultiHeadAttentionBlock

class GptBlock(nn.Module):
    def __init__(self, head_size: int, max_seq_length: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.sa = MultiHeadAttentionBlock(head_size, max_seq_length, n_heads, dropout)
        self.ff = FeedForward(head_size, dropout)
        self.ln1 = nn.LayerNorm(head_size)
        self.ln2 = nn.LayerNorm(head_size)
        
    def forward(self, x) -> None:
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
        