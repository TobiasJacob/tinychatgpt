import torch
import torch.nn as nn
import torch.nn.functional as F

from tinychatgpt.models.blocks.SelfAttention import SelfAttentionBlock

class SelfAttentionModel(nn.Module):
    def __init__(self, n_tokens: int, n_embed: int, head_size: int, seq_len: int):
        super(SelfAttentionModel, self).__init__()
        self.encoder = nn.Embedding(n_tokens, n_embed)
        self.attention1 = SelfAttentionBlock(n_embed, head_size, seq_len)
        self.attention2 = SelfAttentionBlock(head_size, head_size, seq_len)
        self.attention3 = SelfAttentionBlock(head_size, head_size, seq_len)
        self.decoder = nn.Linear(head_size, n_tokens)

    def forward(self, x, y=None):
        # x.shape = (batch_size, seq_len)
        # y.shape = (batch_size, seq_len)
        x = self.encoder(x)
        # x.shape = (batch_size, seq_len, n_embed)
        x = self.attention1(x)
        x = self.attention2(x)
        x = self.attention3(x)
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
