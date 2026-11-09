import json
import os
import re
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from maths_solver import try_solve_maths
from model import MiniGPT
from tokenizer import decode, encode, load_vocab, tokenize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pt")
STATIC_DIR = os.path.join(BASE_DIR, "static")

MAX_INPUT_CHARS = 200
MAX_NEW_TOKENS = 24
TEMPERATURE = 0.3
FALLBACK_REPLY = "Sorry, I do not know that yet."

runtime = {"model": None, "stoi": None, "itos": None}


def load_model():
    if not os.path.exists(MODEL_PATH):
        print("model.pt not found. Run 'python train.py' first.")
        return
    stoi, itos = load_vocab()
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model = MiniGPT(checkpoint["vocab_size"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    runtime.update(model=model, stoi=stoi, itos=itos)


@asynccontextmanager
async def lifespan(app):
    load_model()
    yield


app = FastAPI(lifespan=lifespan)


class ChatIn(BaseModel):
    text: str


def clean_text(text):
    return " ".join(re.sub(r"[^a-z0-9' ]", " ", text.lower()).split())


def is_known(text):
    tokens = tokenize(clean_text(text))
    return bool(tokens) and all(tok in runtime["stoi"] for tok in tokens)


def ask_model(text):
    prompt = f"KRB: {clean_text(text)}\nVECTOR:"
    ids = torch.tensor([encode(prompt, runtime["stoi"])], dtype=torch.long)
    out = runtime["model"].generate(ids, MAX_NEW_TOKENS, temperature=TEMPERATURE)[0].tolist()
    reply = decode(out, runtime["itos"]).split("VECTOR:", 1)[-1]
    reply = reply.split("\n")[0].split("KRB:")[0]
    return reply.replace("<unk>", "").strip()


def ndjson(payload):
    return Response(json.dumps(payload) + "\n", media_type="application/x-ndjson")


@app.get("/api/health")
def health():
    return {"llm": runtime["model"] is not None}


@app.post("/api/chat")
def chat(body: ChatIn):
    text = body.text.strip()[:MAX_INPUT_CHARS]
    solved = try_solve_maths(text)
    if solved is not None:
        return ndjson({"t": "tok", "v": solved})
    if runtime["model"] is None:
        return ndjson({"t": "err", "v": "Model is not loaded yet."})
    if not is_known(text):
        return ndjson({"t": "tok", "v": FALLBACK_REPLY})
    reply = ask_model(text) or FALLBACK_REPLY
    return ndjson({"t": "tok", "v": reply})


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
