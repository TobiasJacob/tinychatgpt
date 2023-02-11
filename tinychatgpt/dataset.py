import os
from datasets import load_dataset
import random
from transformers import BertTokenizer
from tqdm import tqdm
import torch
from multiprocessing.pool import ThreadPool

PATH = os.path.join(os.path.dirname(__file__), "dataset.txt")

def generate_dataset():
    dataset = load_dataset("amazon_reviews_multi", "de")
    train = dataset["train"]

    lines = []
    with open("dataset.txt", "w", encoding="utf-8") as f:
        for i, r in enumerate(train):
            if len(r['review_body']) > 300:
                lines.append(f"Stars: {r['stars']}; Review Body: {r['review_body']}\n")
            if len(lines) > 30:
                f.writelines(
                    lines
                )
                lines = []

CHARACTERS = set("\n !(),.0123456789qwertyuioplkjhgfdsazxcvbnmQWWERTYUIOPLKJHGFDSAZXCVBNMäöüÄÖÜß")
CHARACTERS.add("Stars: ")
CHARACTERS.add("; Review Body: ")

STOI = {char: i for i, char in enumerate(CHARACTERS)}
ITOS = {i: char for i, char in enumerate(CHARACTERS)}

TOKENIZER = BertTokenizer.from_pretrained("bert-base-german-dbmdz-cased")
TOKENIZER.add_tokens(["Stars: ", "; Review Body: "])
MODE = "bert"

if MODE == "bert":
    N_TOKENS = TOKENIZER.vocab_size + 2
else:
    N_TOKENS = len(CHARACTERS)

print(f"Using {MODE} tokenizer with {N_TOKENS} tokens")

def encode(text):
    if MODE == "bert":
        return TOKENIZER.encode(text, add_special_tokens=False)
    return [STOI[char] for char in text if char in CHARACTERS]

def decode(text):
    if MODE == "bert":
        return TOKENIZER.decode(text)
    return "".join([ITOS[char] for char in text])


CACHE_PATH = os.path.join(os.path.dirname(__file__), f"dataset_cache_{TOKENIZER.__class__.__name__}.torch")

def load_dataset():
    if os.path.exists(CACHE_PATH):
        return torch.load(CACHE_PATH)

    if not os.path.exists(PATH):
        generate_dataset()
    
    with open(PATH, "r", encoding="utf-8") as f:
        dataset = f.readlines()    

    tokenized_dataset = []
    # Tokenize dataset in parallel
    with tqdm(total=len(dataset)) as pbar:
        def update(*a):
            pbar.update()

        def tokenize_line(line):
            if MODE == "bert":
                return TOKENIZER.encode(line, add_special_tokens=False)
            else:
                # Parse stars
                stars = line.split("Stars: ")[1].split(";")[0]
                # Parse review body
                review = line.split("; Review Body: ")[1]

                return [STOI["Stars: "]] + encode(stars) + [STOI["; Review Body: "]] + encode(review)

        with ThreadPool(16) as p:
            for line in p.imap_unordered(tokenize_line, dataset, chunksize=10):
                tokenized_dataset.append(line)
                update()

    # shuffle dataset
    random.shuffle(tokenized_dataset)

    tokenized_dataset = [char for line in tokenized_dataset for char in line]

    tokenized_dataset = torch.tensor(tokenized_dataset, dtype=torch.long)
    torch.save(tokenized_dataset, CACHE_PATH)

    return tokenized_dataset
