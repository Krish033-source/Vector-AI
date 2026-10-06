# Vector

A GPT-style transformer language model, **built and trained from scratch in PyTorch** — no pretrained weights, no HuggingFace `transformers`, no external API calls. Every parameter was randomly initialized and learned entirely from a ~27KB hand-written dataset. Talks back by text or voice, and solves arithmetic through a deterministic hybrid solver rather than pretending a 5M-parameter model can "do maths" on its own.

Architecture style is inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) — decoder-only, causal self-attention, next-token prediction — reimplemented with a custom digit-aware tokenizer, a persona/fact dataset, and a hybrid neural+symbolic inference pipeline.

---

## Maths — 100% accurate (by design, not by training)

The neural network is genuinely bad at arithmetic — this is a real, well-documented limitation of small transformers, not a bug:

| Operation | Neural net accuracy (asked directly) | Hybrid solver accuracy |
|---|---|---|
| Addition | 7.7% (13/169) | **100%** |
| Subtraction | 11.0% (10/91) | **100%** |
| Multiplication | 26.0% (26/100) | **100%** |
| **Overall** | **13.6% (49/360)** | **100%** |

Instead of chasing marginally better memorization, `maths_solver.py` detects arithmetic expressions — digits **or** spoken words ("two plus three" included, since speech-to-text doesn't always convert numbers) — and computes them directly in Python:

```
> 7 + 5          > 23 - 9         > 6 * 7          > nine times eight
12               14               42               72
```

This is the same "tool use" principle production AI systems rely on: a model doesn't need arithmetic baked into its weights if it knows when to call a calculator instead.

---

## What it can do

- Answer **identity/personality questions** ("who are you", "what is your name", "who made you") reliably
- Answer a small set of **memorized facts** about its creator's tech stack and general knowledge
- Do **exact arithmetic** via the hybrid solver above
- Run as a **voice assistant**: wake word ("Vector") → speech-to-text → model/solver → text-to-speech
- Run entirely on **CPU** — optional GPU acceleration via CUDA (NVIDIA) or DirectML (AMD/Intel on Windows)

---

## Built with

| Component | Tool |
|---|---|
| Model + training | PyTorch (`torch.nn`, `torch.optim.AdamW`, autograd) — no `transformers`, no pretrained checkpoints |
| Tokenizer | Custom word-level tokenizer (`re`), with digit-level splitting |
| Arithmetic | Pure Python + regex (`maths_solver.py`) — computed, not learned |
| Voice I/O | `SpeechRecognition` (Google Web Speech API) + `pyttsx3` (offline TTS) |
| GPU accel. (optional) | CUDA (NVIDIA) or DirectML (`torch-directml`, AMD/Intel on Windows) |

---

## Architecture & the math

```
tokens → token embedding + positional embedding
       → [ Multi-Head Self-Attention → Add & Norm → FeedForward → Add & Norm ] × 6
       → final LayerNorm
       → Linear → vocab logits
```

**Self-attention:**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

$d_k$ is the per-head dimension (scaling by $\sqrt{d_k}$ keeps dot-product magnitudes stable before softmax); $M$ is a causal mask so position $i$ can never attend to positions $> i$.

**Multi-head attention** runs this in parallel across 8 heads, concatenates the results, and projects back to the embedding dimension.

**Loss** — next-token cross-entropy:

$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log P(x_t \mid x_{<t})$$

**Perplexity** ($e^{\mathcal{L}}$) at the current checkpoint (validation loss 0.78): **≈ 2.19** — on held-out data the model is, on average, about as uncertain as choosing between ~2 roughly-equally-likely tokens. For scale: a uniform random guess over the 144-token vocabulary would be perplexity 144.

### Hyperparameters

| | |
|---|---|
| Embedding dimension | 256 |
| Attention heads | 8 |
| Transformer layers | 6 |
| Context window | 128 tokens |
| Vocabulary size | 144 tokens |
| **Total parameters** | **4,841,104** |
| Optimizer | AdamW, lr 3e-4 |
| Dropout | 0.1 |

### Tokenizer

Word-level tokenization with digits split individually (`"23"` → `"2"`, `"3"`) instead of treated as one opaque token — giving the model at least a structural chance to learn arithmetic as a digit pattern. In practice, a model this size still can't make reliable use of that structure (see the maths table above), which is exactly why arithmetic is handled by a separate deterministic solver.

---

## Accuracy

Measured with `eval_accuracy.py` — every fact/identity question in the dataset actually queried and checked, not eyeballed:

| Category | Accuracy |
|---|---|
| Identity questions | **92.1%** (35/38) |
| Tech-stack facts | **77.8%** (14/18) |
| General facts | **69.2%** (9/13) |
| Arithmetic | **100%** (via hybrid solver — see above) |

**Caveat:** `generate.py` samples with temperature 0.8 by default (for more natural output), so the same question can get different answers on different runs. The table above uses near-greedy decoding for a consistent read of what the model has actually learned. For a reliable live demo, lower `temperature` in the `generate()` calls (e.g. to 0.3).

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
```

Captured from an actual run on this checkpoint.

---

## Limitations

- This is memorization with some paraphrase-robustness, not general language understanding. Reword a question enough and it will confidently answer something unrelated rather than failing gracefully.
- ~5M parameters trained from scratch on ~27KB of text cannot learn robust arithmetic through next-token prediction alone — a known limitation of small transformers, not a bug. That's why `maths_solver.py` exists.
- Inputs are lowercased before encoding (training data is all lowercase), but sentence-structure sensitivity remains — see the accuracy caveat above.

---

## Setup

```bash
pip install torch
# optional, Windows + AMD/Intel GPU only:
# pip install torch-directml

python build_dataset.py     # generates dataset.txt
python train.py 5000        # trains model.pt (resumable)
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
| `model.py` | Transformer architecture |
| `tokenizer.py` | Word-level tokenizer with digit-splitting and hyphen-aware decoding |
| `build_dataset.py` | Generates the training dataset |
| `train.py` | Training loop — resumable, early-stopping, CUDA/DirectML/CPU-aware |
| `generate.py` | CLI inference |
| `voice_assistant.py` | Mic → wake word → model/solver → speech |
| `maths_solver.py` | Deterministic arithmetic (digits and spoken number words) |
| `eval_accuracy.py` | Measures real accuracy across the full dataset |
