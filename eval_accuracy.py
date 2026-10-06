import os
import re
import torch

from tokenizer import load_vocab, encode, decode
from model import MiniGPT
from build_dataset import (
    identity_pairs, tech_pairs, fact_pairs, MATHS_RANGE,
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    stoi, itos = load_vocab()
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model = MiniGPT(checkpoint["vocab_size"]).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, stoi, itos


def ask(model, stoi, itos, question, max_new_tokens=12):
    prompt = f"KRB: {question}\nVECTOR:"
    context = torch.tensor([encode(prompt, stoi)], dtype=torch.long).to(DEVICE)
    out = model.generate(context, max_new_tokens=max_new_tokens, temperature=0.0 + 1e-8)[0].tolist()
    text = decode(out, itos)

    after = text.split("VECTOR:", 1)[-1]
    cut_pos = after.find("KRB:")
    if cut_pos != -1:
        after = after[:cut_pos]
    after = after.split("\n")[0].strip()
    after = after.replace("<unk>", "").strip()
    return after


def check_maths(model, stoi, itos):
    results = {"+": [0, 0], "-": [0, 0], "*": [0, 0]}
    wrong_examples = []

    for a in range(0, MATHS_RANGE + 1):
        for b in range(0, MATHS_RANGE + 1):
            q, correct, op = f"{a} + {b}", str(a + b), "+"
            pred = ask(model, stoi, itos, q)
            results[op][1] += 1
            if pred == correct:
                results[op][0] += 1
            elif len(wrong_examples) < 15:
                wrong_examples.append((q, correct, pred))

    for a in range(0, MATHS_RANGE + 1):
        for b in range(0, a + 1):
            q, correct, op = f"{a} - {b}", str(a - b), "-"
            pred = ask(model, stoi, itos, q)
            results[op][1] += 1
            if pred == correct:
                results[op][0] += 1
            elif len(wrong_examples) < 15:
                wrong_examples.append((q, correct, pred))

    for a in range(0, 10):
        for b in range(0, 10):
            q, correct, op = f"{a} * {b}", str(a * b), "*"
            pred = ask(model, stoi, itos, q)
            results[op][1] += 1
            if pred == correct:
                results[op][0] += 1
            elif len(wrong_examples) < 15:
                wrong_examples.append((q, correct, pred))

    return results, wrong_examples


def check_qa(model, stoi, itos, pairs, label):
    correct_count = 0
    wrong_examples = []
    for q, correct in pairs:
        pred = ask(model, stoi, itos, q)
        is_correct = correct.lower() in pred.lower() or pred.lower() in correct.lower()
        if is_correct:
            correct_count += 1
        else:
            wrong_examples.append((q, correct, pred))
    return correct_count, len(pairs), wrong_examples


def main():
    print(f"Loading model from {MODEL_PATH} ...")
    model, stoi, itos = load_model()

    print("\n" + "=" * 60)
    print("MATHS ACCURACY")
    print("=" * 60)
    maths_results, maths_wrong = check_maths(model, stoi, itos)
    total_correct = sum(v[0] for v in maths_results.values())
    total_count = sum(v[1] for v in maths_results.values())
    for op, (correct, count) in maths_results.items():
        pct = 100 * correct / count if count else 0
        print(f"  {op}  : {correct:4d} / {count:4d}  ({pct:5.1f}%)")
    print(f"  TOTAL maths accuracy: {total_correct}/{total_count} ({100*total_correct/total_count:.1f}%)")

    if maths_wrong:
        print("\n  Sample wrong answers:")
        for q, correct, pred in maths_wrong[:10]:
            print(f"    {q}  -> got {pred!r}, expected {correct!r}")

    print("\n" + "=" * 60)
    print("IDENTITY / FACTS ACCURACY")
    print("=" * 60)
    for label, pairs in [("identity", identity_pairs), ("tech", tech_pairs), ("facts", fact_pairs)]:
        correct, total, wrong = check_qa(model, stoi, itos, pairs, label)
        pct = 100 * correct / total if total else 0
        print(f"  {label:10s}: {correct:3d} / {total:3d}  ({pct:5.1f}%)")
        for q, c, p in wrong[:5]:
            print(f"      wrong: {q!r} -> got {p!r}, expected {c!r}")


if __name__ == "__main__":
    main()
