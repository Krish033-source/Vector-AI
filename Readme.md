# Vector

A GPT-style transformer language model, **built and trained from scratch in PyTorch** — no pretrained weights, no HuggingFace `transformers`, no OpenAI/Anthropic API calls. Every matrix in it was initialized randomly and learned entirely from a ~27KB hand-written dataset. It talks back by text or by voice, and solves arithmetic through a deterministic hybrid solver rather than pretending a 5M-parameter model can "do maths" on its own.

Architecture style is inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) — decoder-only, causal self-attention, next-token prediction — reimplemented and adapted here with a custom digit-aware tokenizer, a persona/fact dataset, and a hybrid neural+symbolic inference pipeline.

---

## What it can actually do

- Answer **identity/personality questions** ("who are you", "what is your name", "who made you") reliably
- Answer a small set of **memorized facts** about its creator's tech stack and general knowledge
- Do **exact arithmetic** (addition, subtraction, multiplication — digits or spoken words like "two plus three") via a rule-based solver, not the neural net
- Run as a **voice assistant**: wake word ("Vector") → speech-to-text → model/solver → text-to-speech
- All of this on **CPU**, no GPU required (optional AMD/Intel GPU acceleration via DirectML on Windows, or CUDA if you have NVIDIA)

**What it can't do**, by design, and why that's an honest trade rather than a bug — see [Limitations](#limitations).

---

## Built with

| Component | Tool |
|---|---|
| Model + training | PyTorch (`torch.nn`, `torch.optim.AdamW`, autograd) — no `transformers`, no pretrained checkpoints |
| Tokenizer | Custom word-level tokenizer, hand-written in Python (`re`), with digit-level splitting |
| Arithmetic | Pure Python + regex (`maths_solver.py`) — not learned, computed |
| Voice I/O | `SpeechRecognition` (Google Web Speech API) + `pyttsx3` (offline TTS) |
| GPU accel. (optional) | CUDA (NVIDIA) or DirectML (`torch-directml`, AMD/Intel on Windows) |

---

## Architecture & the math

Decoder-only transformer, same family as GPT-2/GPT-3, scaled down:

```
tokens → token embedding + positional embedding
       → [ Multi-Head Self-Attention → Add & Norm → FeedForward → Add & Norm ] × 6
       → final LayerNorm
       → Linear → vocab logits
```

**Self-attention** — for each head, queries/keys/values are linear projections of the input, and attention weights are:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

where $d_k$ is the per-head dimension (scaling by $\sqrt{d_k}$ keeps dot-product magnitudes stable before softmax) and $M$ is a causal mask (upper-triangular $-\infty$) so position $i$ can never attend to positions $> i$ — the model can only use past tokens to predict the next one.

**Multi-head attention** runs this in parallel across 8 heads with smaller $d_k$ each, concatenates the results, and projects back to the embedding dimension — letting different heads specialize in different kinds of relationships between tokens.

**Loss** — standard next-token cross-entropy over the vocabulary:

$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log P(x_t \mid x_{<t})$$

**Perplexity** (a more interpretable readout of loss — "how many tokens was the model effectively choosing between, on average"):

$$\text{Perplexity} = e^{\mathcal{L}}$$

At the current checkpoint (validation loss 0.78): **perplexity ≈ 2.19** — on held-out data, the model is on average about as uncertain as choosing between ~2 roughly-equally-likely tokens. For reference, a model that's just guessing uniformly over the 144-token vocabulary would have perplexity 144; GPT-2 on general web text sits around 20-35 perplexity (on a vastly larger, more diverse vocabulary and task, so not directly comparable — included for scale only).

### Hyperparameters

| | |
|---|---|
| Embedding dimension | 256 |
| Attention heads | 8 |
| Transformer layers | 6 |
| Context window (block size) | 128 tokens |
| Vocabulary size | 144 tokens |
| **Total parameters** | **4,841,104** |
| — of which embeddings (token + positional) | 69,632 |
| — of which output head | 37,008 |
| — of which attention + feedforward blocks (6 layers) | ~4.73M |
| Optimizer | AdamW, lr 3e-4 |
| Dropout | 0.1 |

### Tokenizer design

Word-level tokenization with one deliberate twist: **digits are split individually** (`"23"` → tokens `"2"`, `"3"`) instead of treated as one opaque word-token. This gives the model at least a structural chance to learn arithmetic as a pattern over digits rather than memorizing whole numbers as unrelated vocabulary — though in practice (see accuracy table below) a model this size still can't make reliable use of that structure, which is exactly why arithmetic is handled by a separate deterministic solver rather than the network.

---

## Honest accuracy

Measured with `eval_accuracy.py` — every arithmetic combination and every fact/identity question in the dataset actually queried and checked, not eyeballed:

| Category | Accuracy |
|---|---|
| Identity questions ("who are you", greetings, etc.) | **92.1%** (35/38) |
| Tech-stack facts | **77.8%** (14/18) |
| General facts | **69.2%** (9/13) |
| Arithmetic, asked of the neural net directly | **13.6%** (49/360) |
| Arithmetic, through the hybrid solver (`maths_solver.py`) | **100%** |

**Important caveat the table doesn't show:** `generate.py` samples with temperature 0.8 by default (for more natural, less repetitive output), so answers to the *same* question vary run to run — including sometimes flipping from correct to wrong on facts the model otherwise "knows." The accuracy numbers above were measured with near-greedy decoding for a consistent read on what the model has actually learned; a live demo run can look less reliable than that table suggests purely due to sampling randomness. For a demo, consider lowering `temperature` in `generate()`/`generate_reply()` calls (e.g. to 0.3) for more consistent answers at the cost of less varied phrasing.

---

## Example conversation

```
> KRB: who are you
VECTOR: I am Vector

> KRB: who made you
VECTOR: Krishna made me

> KRB: what is blackvault
VECTOR: An encryption vault

> KRB: what is gitsetlive
VECTOR: A GitHub AI agent

> KRB: what language do you use
VECTOR: Python

> KRB: who is kalpana chawla
VECTOR: An astronaut

> 7 + 5
12

> twelve minus four
8

> 6 * 7
42
```

(Captured from an actual run of `generate.py` / `maths_solver.py` on this checkpoint — not hand-written.)

---

## Limitations

This is **memorization with a little paraphrase-robustness**, not general language understanding — worth being upfront about, especially before describing it anywhere as "an LLM":

- Only works reliably on facts/phrasings close to what's in `dataset.txt`. Reword a question enough and it will confidently answer something unrelated rather than failing gracefully.
- ~5M parameters trained from scratch on ~27KB of text cannot learn robust arithmetic through next-token prediction alone — this is a known limitation of small transformers (and, to a lesser degree, even much larger LLMs without tool use), not a bug. That's the entire reason `maths_solver.py` exists.
- Case-sensitive quirks are handled (inputs are lowercased before encoding), but sentence-structure sensitivity is not — see the accuracy caveat above.

## Setup

```bash
pip install torch
# optional, Windows + AMD/Intel GPU only:
# pip install torch-directml

python build_dataset.py     # generates dataset.txt
python train.py 5000        # trains model.pt (resumable - see train.py --help logic)
python generate.py "KRB: who are you"
```

Voice assistant:
```bash
pip install SpeechRecognition pyttsx3 pyaudio
python voice_assistant.py
```

## Project structure

| File | Purpose |
|---|---|
| `model.py` | Transformer architecture (attention, blocks, embeddings) |
| `tokenizer.py` | Word-level tokenizer with digit-splitting and hyphen-aware decoding |
| `build_dataset.py` | Generates the training dataset |
| `train.py` | Training loop — resumable, early-stopping, CUDA/DirectML/CPU-aware |
| `generate.py` | CLI inference |
| `voice_assistant.py` | Mic → wake word → model/solver → speech |
| `maths_solver.py` | Deterministic arithmetic (digits and spoken number words) |
| `eval_accuracy.py` | Measures real accuracy across the full dataset |
