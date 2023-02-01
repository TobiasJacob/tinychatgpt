import torch
import torch.nn as nn
import torch.nn.functional as F

class NgramModel(nn.Module):
    def __init__(self, n_tokens: int, n_embed: int, n_hidden: int, seq_len: int):
        super(NgramModel, self).__init__()
        self.encoder = nn.Embedding(n_tokens, n_embed)
        self.fc1 = nn.Linear(n_embed * seq_len, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_tokens)
        self.decoder = nn.Linear(n_hidden, n_tokens)

    def forward(self, x, y=None):
        # x.shape = (batch_size, seq_len)
        # y.shape = (batch_size, seq_len)
        x = self.encoder(x[:, :-1])
        # x.shape = (batch_size, seq_len, n_embed)
        x = x.view(x.shape[0], -1)
        # x.shape = (batch_size, seq_len * n_embed)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x.shape = (batch_size, n_tokens)
        if y is not None:
            return F.cross_entropy(x, y[:, -1])
        return x

    def generate(self, x, num_chars: int):
        # x.shape = (batch_size, seq_len)
        for i in range(num_chars):
            x_next = self.forward(x[:, i:])
            # x_next.shape = (batch_size, n_tokens)
            # sample next character
            x_next = torch.multinomial(F.softmax(x_next, dim=1), 1)
            # x_next.shape = (batch_size, 1)
            x = torch.cat((x, x_next), dim=1)
        return x
