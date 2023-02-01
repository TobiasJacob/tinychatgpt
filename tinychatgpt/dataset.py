import os
from datasets import load_dataset
import random

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
N_TOKENS = len(CHARACTERS)

def encode(text):
    return [STOI[char] for char in text if char in CHARACTERS]

def decode(text):
    return "".join([ITOS[char] for char in text])

def load_dataset():
    if not os.path.exists(PATH):
        generate_dataset()
    
    with open(PATH, "r", encoding="utf-8") as f:
        dataset = f.readlines()    

    tokenized_dataset = []
    for i, line in enumerate(dataset):
        # Parse stars
        stars = line.split("Stars: ")[1].split(";")[0]
        # Parse review body
        review = line.split("; Review Body: ")[1]

        tokenized_dataset.append([STOI["Stars: "]] + encode(stars) + [STOI["; Review Body: "]] + encode(review))

    # shuffle dataset
    random.shuffle(tokenized_dataset)

    tokenized_dataset = [char for line in tokenized_dataset for char in line]

    return tokenized_dataset
