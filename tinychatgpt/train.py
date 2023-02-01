#%%
import os
import random
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
from tinychatgpt.models.MultiHeadAttentionModel import MultiHeadAttentionModel

from tinychatgpt.models.SelfAttentionModel import SelfAttentionModel

if ".." not in sys.path:
    sys.path.append("..")

from tinychatgpt.dataset import N_TOKENS, encode, decode, load_dataset
from tinychatgpt.models.NgramModel import NgramModel

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
    start = torch.randint(0, len(split) - seq_len, (batch_size,))
    batch = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
    for i in range(batch_size):
        batch[i] = split[start[i]:start[i] + seq_len]
    return batch

get_batch(train, 32, 10)
# %%

# Define ngram model
# train the model
seq_len = 128
batch_size = 64
n_blocks = 6
n_embed = 384
model = MultiHeadAttentionModel(N_TOKENS, n_embed, n_blocks, seq_len, dropout=0.2, n_heads=6)
print("Model has {} parameters".format(sum(p.numel() for p in model.parameters())))
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
losses_train = []
losses_test = []
trains_per_test = 4
os.makedirs("models", exist_ok=True)
for i in tqdm(range(5000)):
    split = train if i % trains_per_test != 0 else test
    batch = get_batch(split, batch_size, seq_len + 1)
    if split is train:
        model.train()
    else:
        model.eval()
    loss = model(batch[:, :-1], batch[:, 1:])
    if split is train:
        losses_train.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    else:
        for _ in range(trains_per_test - 1):
            losses_test.append(loss.item())
    if i % 100 == 1:
        print(f"Loss train: {losses_train[-1]} Loss test: {losses_test[-1]}")
        torch.save(model.state_dict(), f"models/model{i}.pt")
plt.plot(losses_train)
plt.plot(losses_test)
plt.show()
# %%

# generate text
batch = get_batch(test, 1, seq_len)
model.eval()
print(decode(model.generate(batch, 1000).squeeze().tolist()))

# %%
