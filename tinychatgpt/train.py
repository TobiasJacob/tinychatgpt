#%%
import os
import random
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys

if ".." not in sys.path:
    sys.path.append("..")

from tinychatgpt.models.SelfAttentionModel import SelfAttentionModel
from tinychatgpt.models.MultiHeadAttentionModel import MultiHeadAttentionModel
from tinychatgpt.dataset import N_TOKENS, encode, decode, load_dataset
from tinychatgpt.models.NgramModel import NgramModel

# use SummaryWriter from pytorch to log training
from torch.utils.tensorboard import SummaryWriter

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
# seq_len = 13
# batch_size = 11
# n_blocks = 7
# n_heads = 5
# n_embed = n_heads*3
seq_len = 128
batch_size = 64
n_blocks = 6
n_embed = 384
n_heads = 6

model = MultiHeadAttentionModel(N_TOKENS, n_embed, n_blocks, seq_len, dropout=0.2, n_heads=n_heads)
print("Model has {} parameters".format(sum(p.numel() for p in model.parameters())))
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
trains_per_test = 4

writer = SummaryWriter()
os.makedirs("models", exist_ok=True)

#%%
for i in tqdm(range(0, 5000)):
    split = train if i % trains_per_test != 0 else test
    batch = get_batch(split, batch_size, seq_len + 1)
    if split is train:
        model.train()
    else:
        model.eval()
    loss = model(batch[:, :-1], batch[:, 1:])
    if split is train:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar("Loss/train", loss.item(), i)
    else:
        writer.add_scalar("Loss/eval", loss.item(), i)
    if i % 100 == 1:
        torch.save(model.state_dict(), f"{writer.log_dir}/model{i}.pt")
        # generate text and save it
        batch = get_batch(test, 1, seq_len)
        model.eval()
        text = decode(model.generate(batch, 1000).squeeze().tolist())
        with open(f"{writer.log_dir}/model{i}.txt", "w") as f:
            f.write(text)
    
# %%

# generate text
batch = get_batch(test, 1, seq_len)
model.eval()
print(decode(model.generate(batch, 1000).squeeze().tolist()))

# %%
