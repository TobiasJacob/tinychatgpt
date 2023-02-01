#%%
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import sys

if ".." not in sys.path:
    sys.path.append("..")

from tinychatgpt.dataset import N_TOKENS, encode, decode, load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%
tokenized_dataset = load_dataset()

# %%
decode(tokenized_dataset[:1000])
# %%
train = tokenized_dataset[:int(len(tokenized_dataset) * 0.8)]
test = tokenized_dataset[int(len(tokenized_dataset) * 0.8):]

train = torch.tensor(train, dtype=torch.long).to(device)
test = torch.tensor(test, dtype=torch.long).to(device)

def get_batch(split, batch_size, seq_len):
    start = torch.randint(0, len(split) - seq_len - 1, (batch_size,))
    batch = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
    target = torch.zeros((batch_size), dtype=torch.long, device=device)
    for i in range(batch_size):
        batch[i] = split[start[i]:start[i] + seq_len]
        target[i] = split[start[i] + seq_len]
    return batch, target

get_batch(train, 32, 10)
# %%
# Define ngram model
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
        x = self.encoder(x)
        # x.shape = (batch_size, seq_len, n_embed)
        x = x.view(x.shape[0], -1)
        # x.shape = (batch_size, seq_len * n_embed)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x.shape = (batch_size, n_tokens)
        if y is not None:
            return F.cross_entropy(x, y)
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

# train the model
seq_len = 16
batch_size = 32
model = NgramModel(n_tokens, 256, 256, seq_len)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
losses_train = []
losses_test = []
for i in range(600):
    for split in [train, test]:
        batch, target = get_batch(split, batch_size, seq_len)
        loss = model(batch, target)
        if split is train:
            losses_train.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            losses_test.append(loss.item())
        if i % 100 == 1:
            print(f"Loss train: {losses_train[-1]} Loss test: {losses_test[-1]}")
plt.plot(losses_train)
plt.plot(losses_test)
plt.show()
# %%

# generate text
batch, target = get_batch(test, 1, seq_len)
print(decode(model.generate(batch, 1000).squeeze().tolist()))

# %%
