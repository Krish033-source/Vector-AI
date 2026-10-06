import torch
import os
import sys
import json

from tokenizer import build_vocab, save_vocab, encode
from model import MiniGPT, BLOCK_SIZE

torch.set_num_threads(os.cpu_count())

BATCH_SIZE = 16
MAX_ITERS = 5000
EVAL_INTERVAL = 200
LEARNING_RATE = 3e-4
PATIENCE = 8


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        return torch.device("cpu")


DEVICE = get_device()
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pt")
STATE_PATH = os.path.join(os.path.dirname(__file__), "train_state.json")

FRESH_START = "--fresh" in sys.argv
_cli_args = [a for a in sys.argv[1:] if a != "--fresh"]
if _cli_args:
    MAX_ITERS = int(_cli_args[0])

torch.manual_seed(1337)


def get_batch(data):
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_iters=50):
    out = {}
    model.eval()
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    print(f"Using device: {DEVICE}")

    text, chars, stoi, itos = build_vocab()
    save_vocab(stoi, itos)
    vocab_size = len(chars)
    print(f"Dataset: {len(text)} characters, vocab size: {vocab_size}")

    data = torch.tensor(encode(text, stoi), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    model = MiniGPT(vocab_size).to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    best_val_loss = float("inf")
    bad_evals = 0
    if os.path.exists(MODEL_PATH) and not FRESH_START:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        if checkpoint.get("vocab_size") == vocab_size:
            try:
                model.load_state_dict(checkpoint["model_state_dict"])
                best_val_loss = checkpoint.get("val_loss", float("inf"))
                if os.path.exists(STATE_PATH):
                    with open(STATE_PATH) as f:
                        bad_evals = json.load(f).get("bad_evals", 0)
                print(f"Resumed from checkpoint (val loss so far: {best_val_loss:.4f}, "
                      f"bad_evals: {bad_evals}/{PATIENCE})")
            except RuntimeError:
                print("Checkpoint architecture does not match current model.py "
                      "hyperparameters (e.g. N_EMBD/N_LAYER changed) - starting fresh.")
        else:
            print("Checkpoint vocab size mismatch (dataset changed?) - starting fresh.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for iter in range(MAX_ITERS):
        if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
            losses = estimate_loss(model, train_data, val_data)
            train_loss, val_loss = losses["train"], losses["val"]
            print(f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                bad_evals = 0

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "vocab_size": vocab_size,
                        "val_loss": best_val_loss,
                    },
                    MODEL_PATH,
                )
                print(f"  -> new best model saved (val loss {val_loss:.4f})")
            else:
                bad_evals += 1
                print(f"  -> val loss did not improve ({bad_evals}/{PATIENCE})")

            with open(STATE_PATH, "w") as f:
                json.dump({"bad_evals": bad_evals}, f)

            if bad_evals >= PATIENCE:
                print(f"Early stopping: val loss did not improve for {PATIENCE} evals.")
                break

        xb, yb = get_batch(train_data)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(f"Training done. Best model (val loss {best_val_loss:.4f}) saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
