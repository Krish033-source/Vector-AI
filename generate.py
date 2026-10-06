import sys
import os
import torch

from tokenizer import load_vocab, encode, decode
from model import MiniGPT
from maths_solver import try_solve_maths

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pt")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        return torch.device("cpu")


DEVICE = get_device()


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Krishna is"
    max_new_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    maths_query = prompt
    if maths_query.upper().startswith("KRB:"):
        maths_query = maths_query.split(":", 1)[-1]
    maths_answer = try_solve_maths(maths_query)
    if maths_answer is not None:
        print(maths_answer)
        return

    if not os.path.exists(MODEL_PATH):
        print("model.pt not found. Run 'python train.py' first to train the model.")
        return

    stoi, itos = load_vocab()

    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model = MiniGPT(checkpoint["vocab_size"]).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if prompt.upper().startswith("KRB:"):
        tag, _, rest = prompt.partition(":")
        prompt = f"{tag}:{rest.lower()}"

    context = torch.tensor([encode(prompt, stoi)], dtype=torch.long).to(DEVICE)
    generated = model.generate(context, max_new_tokens=max_new_tokens, temperature=0.8)[0].tolist()

    text = decode(generated, itos)

    cut_pos = text.find("KRB:", len(prompt))
    if cut_pos != -1:
        text = text[:cut_pos].strip()

    text = text.replace(" <unk>", "").replace("<unk> ", "").replace("<unk>", "")
    text = " ".join(text.split())

    print(text)


if __name__ == "__main__":
    main()
