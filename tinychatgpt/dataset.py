#%%
from datasets import load_dataset

dataset = load_dataset("amazon_reviews_multi", "de")
train = dataset["train"]

#%%

lines = []
with open("dataset.txt", "w") as f:
    for i, r in enumerate(train):
        if len(r['review_body']) > 300:
            lines.append(f"Stars: {r['stars']}; Review Body: {r['review_body']}\n")
        if len(lines) > 30:
            f.writelines(
                lines
            )
            lines = []
