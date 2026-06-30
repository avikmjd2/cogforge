# cogforge

> A from-scratch deep learning library built on nothing but NumPy — a reverse-mode autograd engine extended all the way to a working GPT.

`cogforge` is a small, readable, educational deep learning framework. At its core is a `Tensor` that records every operation into a computation graph and backpropagates through it (micrograd-style), but unlike a toy autograd it scales up to real architectures: MLPs, RNNs, batch/layer normalization, multi-head attention, and a decoder-only transformer (`GPTV1`) you can actually train and sample from.

There is no C++, no CUDA, no PyTorch — just NumPy and explicit, hand-derived gradients. The goal is to *understand* every gradient that flows, not to be fast.

---

## Table of contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Core concept: the `Tensor`](#core-concept-the-tensor)
- [API reference](#api-reference)
  - [Tensor — autograd engine](#tensor--autograd-engine)
  - [Losses](#losses)
  - [Layers](#layers)
  - [Containers](#containers)
  - [Recurrent](#recurrent)
  - [Sequence-to-sequence](#sequence-to-sequence)
  - [Optimizers](#optimizers)
  - [Models](#models)
- [Worked example: train a char-level GPT](#worked-example-train-a-char-level-gpt)
- [Gotchas](#gotchas)
- [Roadmap](#roadmap)
- [License](#license)

---

## Installation

```bash
pip install cogforge-engine
```

Requires Python 3.8+ and NumPy. That's the only dependency.

The package is organized into two modules:

| Module | Contains |
| --- | --- |
| `cogforge.app` | The autograd engine (`Tensor`) and every building block — layers, optimizers, losses, normalization, attention. |
| `cogforge.models` | Ready-to-use models. Currently `GPTV1`. |

```python
from cogforge.app import Tensor, Linear, Adam, MultiHeadAttention   # building blocks
from cogforge.models import GPTV1                                    # models
```

---

## Quick start

```python
import numpy as np
from cogforge.app import Tensor

# Build a graph
a = Tensor(np.array([2.0, 3.0]))
b = Tensor(np.array([4.0, 5.0]))
c = (a * b).sigmoid().softmax()

# Backpropagate (note the spelling: backwards, with an 's')
c.backwards()

print(a.grad)   # gradient of the output w.r.t. a
```

Every `Tensor` carries a `.data` (the NumPy array), a `.grad` (same shape, accumulates gradients), and a hidden `_backwards` closure that knows how to push gradient to its parents. Calling `.backwards()` on any node runs a topological sort and walks the graph in reverse.

---

## Core concept: the `Tensor`

```python
Tensor(array, children=(), requires_grad=True, typed="compressed")
```

| Argument | Meaning |
| --- | --- |
| `array` | Any array-like; stored as a NumPy array in `.data`. |
| `children` | Parent tensors in the graph (set internally by ops; you rarely pass this). |
| `requires_grad` | Reserved flag (currently informational). |
| `typed` | `"compressed"` → `float32` (default), anything else → `float64`. |

Gradients **accumulate** into `.grad`. Always zero them between optimization steps (the optimizers do this for you via `zero_grad()`).

---

## API reference

### Tensor — autograd engine

**Differentiable operations** (each builds graph and defines its own backward):

| Operation | Notes |
| --- | --- |
| `a + b`, `a - b`, `a * b` | Elementwise, with broadcasting support. |
| `a @ b` | Batched matmul; gradients are correctly un-broadcast. |
| `a[key]` | Indexing/slicing. |
| `.relu()` | |
| `.sigmoid()` | |
| `.tanh()` | |
| `.softmax(axis=-1)` | Numerically stable (max-subtraction). |
| `.view(shape)` | Reshape (handles non-contiguous data). |
| `.flatten()` | Flattens everything after the batch dim → `(B, -1)`. |
| `.flatten_consective(num)` | Groups `num` consecutive timesteps. Expects a 3-D `(B, T, C)` tensor; `T` must be divisible by `num`. |
| `.transpose(axes)` | Permute axes (pass the full permutation tuple). |
| `.masked_fill(mask, value)` | Sets entries where `mask` is `True` to `value` (used for causal attention). |

**Backward pass**

| Method | Notes |
| --- | --- |
| `.backwards()` | **Primary.** Iterative topological sort — safe for deep/long graphs. |
| `.backwards_recursive()` | Legacy recursive version; can hit Python's recursion limit on long sequences. Prefer `.backwards()`. |

**Static helper**

- `Tensor.unbroadcast(grad, shape)` — reduces a broadcasted gradient back to the original parameter shape. Used internally.

---

### Losses

All losses are **classmethods** on `Tensor` and return a scalar loss tensor you call `.backwards()` on. Mind the distinction between losses that take **probabilities** and losses that take **raw logits** — this is the most common mistake.

| Loss | Input expectation | Use when |
| --- | --- | --- |
| `Tensor.cross_entropy_loss(predictions, targets)` | `predictions` are **probabilities** (call `.softmax()` first), `targets` one-hot. | You already have a softmax in your graph. |
| `Tensor.softmax_cross_entropy(scores, targets)` | `scores` are **raw logits**, `targets` one-hot. Softmax is fused inside (stable). Works for 2-D `(B,V)` and 3-D `(B,T,V)`. | Standard classification / LM. **Recommended.** |
| `Tensor.sparse_softmax_cross_entropy(scores, target_ids)` | `scores` raw logits `(B,T,V)`, `target_ids` integers `(B,T)`. | Language modeling — skips building one-hot targets. |
| `Tensor.cross_entropy_loss_masked(predictions, targets, mask)` | Probabilities + per-row `mask` (1 = real, 0 = pad). | Padded batches. |
| `Tensor.softmax_cross_entropy_masked(scores, targets, mask)` | Logits + per-row mask. | Padded batches, fused softmax. |

> ℹ️ `softmax_cross_entropy` and `sparse_softmax_cross_entropy` apply softmax internally. Do **not** pass already-softmaxed values into them. `cross_entropy_loss` is the opposite — it expects probabilities.

---

### Layers

#### `Linear(nin, nout)`
Affine transform `x @ W + b`. He-initialized weights. `.parameters()` → `[W, b]`.

#### `Embedding(vocab_size, embedding_dim)`
Lookup table. Call with an integer index array; backward scatters gradients correctly (uses `np.add.at`, so repeated indices accumulate). `.parameters()` → `[weights]`.

#### `LayerNorm(dim, eps=1e-5)`
Normalizes over the last dimension. Learnable `gamma`/`beta`. `.parameters()` → `[gamma, beta]`.

#### `BatchNorm1D(dim, eps=1e-5, momentum=0.1)`
Normalizes over the batch (and time, for 3-D input). Tracks `running_mean`/`running_var` for inference. Toggle `.training = True/False`. Learnable `gamma`/`beta`.

#### `Attention(dk)`
Scaled dot-product attention. Call `attention(Q, K, V, mask=None)`. `dk` is the key dimension (sets the `1/√dk` scale).

#### `MultiHeadAttention(dinp, dmodel, dout, n)`
`n` heads, `dmodel` split into `n` chunks of size `dmodel // n` (must divide evenly). Projects input `dinp → dmodel`, attends, projects `dmodel → dout`. Call `mha(query, key, value, mask=None)`. `.parameters()` returns all four projection layers' params.

#### `FeedForward(dmodel, dff=None)`
Position-wise MLP: `Linear → ReLU → Linear`. `dff` defaults to `4 * dmodel`. `.parameters()` included.

#### `Transformer(dmodel, n, dff=None)`
A **pre-norm** decoder block: `x + Attn(LN(x))` then `x + FF(LN(x))`. `n` = number of attention heads. Call `block(x, mask=None)`. `.parameters()` included.

#### `PositionalEncoding(max_len, dmodel)`
Fixed sinusoidal positions, added to the input. Call `pe(x)`. No parameters.

---

### Containers

#### `Sequential(layers)`
Runs layers in order. `.train()` / `.test()` flip the `training` flag on any layer that has one (e.g. `BatchNorm1D`).

> ⚠️ `Sequential.parameters()` only collects layers exposing `W`, `b`, `gamma`, or `beta` attributes (i.e. `Linear`, `LayerNorm`, `BatchNorm1D`). Composite layers like `MultiHeadAttention`, `FeedForward`, and `Transformer` hold sub-modules, so their parameters are **not** picked up here — gather those via each module's own `.parameters()`.

#### `MLP(layer_sizes)`
Convenience feed-forward net: `Linear → ReLU` between layers, plain `Linear` output. Built from a list of sizes, e.g. `MLP([784, 128, 64, 10])`.

- `.save(filename="best_model.npz")` / `.load(filename="best_model.npz")` — persist/restore weights.
- *Note:* `MLP` does not expose a `parameters()` method; collect them via `[p for layer in mlp.layers for p in layer.parameters()]` if you want to optimize it.

---

### Recurrent

#### `RNNCell(input_dim, hidden_dim)`
One tanh recurrence step: `h_next = tanh(i2h(x) + h2h(h_prev))`. `.parameters()` included.

#### `RNN(input_dim, hidden_dim)`
Unrolls a cell over a **list** of timestep tensors (each `(B, input_dim)`) and returns the list of hidden states (each `(B, hidden_dim)`). Optional `prev_hidden`.

#### `StackedRNN(input_dim, hidden_dim, num_layers)`
Multiple `RNN` layers stacked. Returns `(top_layer_states, per_layer_final_states)` — the second value is convenient for seq2seq.

---

### Sequence-to-sequence

#### `Bridge(enc_hidden, dec_hidden, enc_layers, dec_layers, mode="project")`
Maps encoder final hidden states to decoder initial hidden states, handling mismatched layer counts and hidden sizes.

| `mode` | Behavior |
| --- | --- |
| `"project"` | One learned `Linear(enc_hidden → dec_hidden)` per decoder layer. General, recommended. |
| `"tie"` | No parameters; requires `enc_hidden == dec_hidden`. Selects/repeats raw states. |

---

### Optimizers

Both take an iterable of parameter tensors and share the same interface: `step()`, `zero_grad()`, `clip_grads(max_norm=5.0)`.

#### `SGD(parameters, learning_rate=0.01)`
Plain stochastic gradient descent.

#### `Adam(parameters, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8)`
Adam with bias correction. Recommended for transformers.

```python
opt = Adam(model.parameters(), lr=3e-4)
opt.zero_grad()
loss.backwards()
opt.clip_grads(1.0)   # optional gradient clipping
opt.step()
```

---

### Models

#### `GPTV1(vocab, d_model, n_heads, n_layers, max_len, d_ff=None)`
A decoder-only transformer (token embedding + sinusoidal positions + stacked pre-norm `Transformer` blocks + final `LayerNorm` + output head). Causal masking is applied internally.

| Method | Description |
| --- | --- |
| `model(idx)` | `idx`: integer array `(B, T)`. Returns logits `(B, T, vocab)`. |
| `model.parameters()` | All trainable tensors. |
| `model.generate(idx, n_new, temperature=1.0, top_k=None)` | Autoregressive sampling. Crops to `max_len`, supports temperature and top-k. Returns `(B, T + n_new)`. |

---

## Worked example: train a char-level GPT

```python
import numpy as np
from cogforge.app import Tensor, Adam
from cogforge.models import GPTV1

# --- data -------------------------------------------------------------
text  = open("input.txt").read()
chars = sorted(set(text))
stoi  = {c: i for i, c in enumerate(chars)}
itos  = {i: c for i, c in enumerate(chars)}
data  = np.array([stoi[c] for c in text])
vocab = len(chars)

# --- model ------------------------------------------------------------
block = 64
model = GPTV1(vocab=vocab, d_model=128, n_heads=4,
              n_layers=4, max_len=block)
opt   = Adam(model.parameters(), lr=3e-4)

def get_batch(bs=32):
    ix = np.random.randint(0, len(data) - block - 1, size=bs)
    x  = np.stack([data[i:i + block]     for i in ix])
    y  = np.stack([data[i + 1:i + block + 1] for i in ix])
    return x, y

# --- train ------------------------------------------------------------
for step in range(2000):
    x, y   = get_batch()
    logits = model(x)                                  # (B, T, vocab)
    loss   = Tensor.sparse_softmax_cross_entropy(logits, y)

    opt.zero_grad()
    loss.backwards()
    opt.clip_grads(1.0)
    opt.step()

    if step % 100 == 0:
        print(f"step {step:4d} | loss {loss.data:.4f}")

# --- sample -----------------------------------------------------------
ctx = np.array([[stoi["\n"]]])
out = model.generate(ctx, n_new=300, temperature=0.8, top_k=20)
print("".join(itos[i] for i in out[0]))
```

---

## Gotchas

- **It's `backwards()`, not `backward()`.** The backward pass method has a trailing `s`.
- **Logits vs. probabilities.** `softmax_cross_entropy` / `sparse_softmax_cross_entropy` fuse the softmax internally — feed them **raw logits**. `cross_entropy_loss` expects **probabilities**. Mixing these up silently trains the wrong thing.
- **Gradients accumulate.** Call `optimizer.zero_grad()` every step (or `p.grad[...] = 0`), or gradients pile up across iterations.
- **`Sequential.parameters()` is shallow** — see the note under [Containers](#containers). For attention/feed-forward/transformer stacks, gather parameters through each module's own `.parameters()` (as `GPTV1.parameters()` does).
- **RNNs operate on lists**, not a single `(B, T, C)` tensor — pass a list of per-timestep tensors.

---

## Roadmap

Planned / under consideration:

- RoPE (rotary position embeddings) with length interpolation
- SwiGLU feed-forward and RMSNorm
- Weight tying between embedding and output head
- KV cache for faster generation
- Linear-attention block (as a study in the recall-vs-cost tradeoff)

---

## License

MIT License. See [LICENSE](LICENSE) for details.