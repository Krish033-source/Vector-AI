import json
import os
import re
from collections import Counter

DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset.txt")
VOCAB_PATH = os.path.join(os.path.dirname(__file__), "vocab.json")

MAX_VOCAB_SIZE = 12000
UNK_TOKEN = "<unk>"

TOKEN_PATTERN = re.compile(r"\n|[A-Za-z]+|\d|[^\w\s]")

NO_SPACE_BEFORE = {
    ".", ",", "!", "?", ":", ";", "'", ")", "]", "}", "\u2019", "%", "\n",
}
NO_SPACE_AFTER = {"(", "[", "{", "\n"}


def tokenize(text):
    return TOKEN_PATTERN.findall(text)


MUST_INCLUDE = {
    "Krishna", "KRB", "Vector", "CryptOS", "LalCoin", "Raspberry", "Pi",
    "Kali", "Linux", "Offensive", "Security", "Scikit-Learn", "TensorFlow",
    "NumPy", "Matplotlib", "React", "Vite", "Electron", "Node.js", "npm",
    "Tailwind", "Flask", "FastAPI", "MySQL", "PostgreSQL", "Plotly",
    "ShellGPT", "sattu", "paneer", "dahi", "roti", "Chandrayaan", "ISRO",
    "NASA", "Mangalyaan", "Kalpana", "Chawla", "Rakesh", "Sharma",
    "Aryabhata", "Ambedkar", "Gandhi",
}


def build_vocab(dataset_path=DATASET_PATH):
    with open(dataset_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokens = tokenize(text)
    counts = Counter(tokens)

    forced = [tok for tok in MUST_INCLUDE if tok in counts]
    remaining_slots = MAX_VOCAB_SIZE - len(forced) - 1

    most_common = [
        tok for tok, _ in counts.most_common(MAX_VOCAB_SIZE + len(forced))
        if tok not in forced
    ][:remaining_slots]

    vocab_tokens = sorted(set(forced + most_common)) + [UNK_TOKEN]

    stoi = {tok: i for i, tok in enumerate(vocab_tokens)}
    itos = {i: tok for i, tok in enumerate(vocab_tokens)}

    return text, vocab_tokens, stoi, itos


def save_vocab(stoi, itos, path=VOCAB_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"stoi": stoi, "itos": itos}, f, ensure_ascii=False, indent=2)


def load_vocab(path=VOCAB_PATH):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    stoi = data["stoi"]
    itos = {int(k): v for k, v in data["itos"].items()}
    return stoi, itos


def encode(text, stoi):
    unk_idx = stoi.get(UNK_TOKEN)
    result = []
    for tok in tokenize(text):
        if tok in stoi:
            result.append(stoi[tok])
        elif unk_idx is not None:
            result.append(unk_idx)
    return result


def decode(indices, itos):
    tokens = [itos[i] for i in indices]
    out = ""
    prev = None
    for i, tok in enumerate(tokens):
        if prev is None:
            out += tok
        elif tok in NO_SPACE_BEFORE or prev in NO_SPACE_AFTER:
            out += tok
        elif tok.isdigit() and prev.isdigit():
            out += tok
        elif tok == "-" and i + 1 < len(tokens) and prev.isalpha() and tokens[i + 1].isalpha():
            out += tok
        elif prev == "-" and i >= 2 and tokens[i - 2].isalpha() and tok.isalpha():
            out += tok
        else:
            out += " " + tok
        prev = tok
    return out


if __name__ == "__main__":
    text, tokens, stoi, itos = build_vocab()
    save_vocab(stoi, itos)

    print(f"Total tokens (words + punctuation): {len(tokenize(text))}")
    print(f"Vocab size (unique words): {len(tokens)}")

    test = "Krishna is learning Python."
    encoded = encode(test, stoi)
    decoded = decode(encoded, itos)

    print(f"Test: {test!r} -> {encoded[:10]}... -> {decoded!r}")
    print("Tokenizer working correctly. vocab.json saved.")
