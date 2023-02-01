import torch
import torch.nn as nn
import torch.nn.functional as F
from tinychatgpt.models.blocks.GptBlock import GptBlock
from tinychatgpt.models.blocks.MultiHeadAttention import MultiHeadAttentionBlock

from tinychatgpt.models.blocks.SelfAttention import SelfAttentionBlock

class MultiHeadAttentionModel(nn.Module):
    def __init__(self, n_tokens: int, n_embed: int, n_blocks: int, seq_len: int, dropout: float = 0.2, n_heads: int = 6):
        super(MultiHeadAttentionModel, self).__init__()
        self.encoder = nn.Embedding(n_tokens, n_embed)
        self.pos_encoder = nn.Embedding(seq_len, n_embed)
        self.blocks = nn.Sequential(*[GptBlock(n_embed, seq_len, n_heads, dropout) for _ in range(n_blocks)])
        self.decoder_norm = nn.LayerNorm(n_embed)
        self.decoder = nn.Linear(n_embed, n_tokens)

    def forward(self, x, y=None):
        # x.shape = (batch_size, seq_len)
        # y.shape = (batch_size, seq_len)
        pos_emb = self.pos_encoder(torch.arange(x.shape[1], device=x.device))
        x = self.encoder(x)
        x = x + pos_emb[None]
        # x.shape = (batch_size, seq_len, n_embed)
        x = self.blocks(x)
        x = self.decoder_norm(x)
        x = self.decoder(x)
        B, T, C = x.shape
        # x.shape = (batch_size, seq_len, n_tokens)
        if y is not None:
            x = x.view(B*T, C)
            y = y.reshape(B*T)
            return F.cross_entropy(x, y)
        return x

    def generate(self, x, num_chars: int):
        # x.shape = (batch_size, seq_len)
        for i in range(num_chars):
            x_next = self.forward(x[:, i:])
            # x_next.shape = (batch_size, n_tokens)
            # sample next character
            x_next = torch.multinomial(F.softmax(x_next[:, -1], dim=1), 1)
            # x_next.shape = (batch_size, 1)
            x = torch.cat((x, x_next), dim=1)
        return x
