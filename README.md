# tinytorch

A from-scratch, scalar reverse-mode **automatic-differentiation engine** that grows
into a small neural-network library, trains an MLP, and is benchmarked against
PyTorch. Built micrograd-style — the goal is *deep, defensible understanding* of how
backpropagation actually works, not a polished clone.

> Status: **M0 — repo scaffolded.** The autodiff core is implemented by hand, one
> gradient-checked operation at a time. See [`CLAUDE.md`](CLAUDE.md) for the full
> milestone plan.

---

## 1. What is autodiff, and why reverse mode?

> _TODO (write-up): the chain rule as graph traversal; forward-record / backward-fire;
> why reverse mode wins for ML — one scalar loss, many parameters, so a single backward
> pass computes every ∂loss/∂param at once (vs. forward mode's one pass per input)._

## 2. Design

> _TODO: the computational graph; the `Value` node (`data`, `grad`, `_backward`,
> `_prev`, `_op`); the three-step op pattern (compute → record → define-closure);
> gradient **accumulation** (`+=` over every use of a node); reverse-topological
> ordering. Include a rendered graph diagram from `viz.draw_graph` (M3)._

![worked-example computational graph](assets/graph_placeholder.png) <!-- TODO (M3) -->

## 3. Results

> _TODO:_
> - Two-moons decision boundary (M4)
> - MNIST accuracy vs. the PyTorch baseline, and an honest note on the speed gap
>   (scalar / NumPy will be far slower — that gap is a teaching result, not a failure) (M7)

---

## Setup

Built and tested with **Python 3.14.0**.

```powershell
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# macOS / Linux
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The PyTorch benchmark baseline (M7 only) is a separate, heavier install:

```
pip install -r requirements-bench.txt
```

> Graph rendering (M3) uses the `graphviz` Python bindings, which call the system
> Graphviz `dot` binary — install Graphviz separately and ensure `dot` is on your PATH.

## Reproducing results

Everything is seeded from a single constant (`tinytorch.SEED`, currently `1337`) via
`tinytorch.seed_everything()`, with deterministic data splits.

```bash
pytest                          # gradient checks + op tests (the correctness oracle)
python examples/train_moons.py  # M4 — two-moons decision boundary
python examples/train_mnist.py  # M7 — MNIST training + accuracy
python benchmarks/pytorch_baseline.py  # M7 — PyTorch comparison
```

> _Most commands above are placeholders until their milestone lands._

## Repo structure

```
tinytorch/
  engine.py        # Value / Tensor + autodiff core   (implemented by hand, M1/M2/M5)
  nn.py            # Neuron/Layer/MLP, then Module/Linear/activations (M4/M6)
  optim.py         # SGD, Adam                          (M4/M6)
  functional.py    # softmax, cross-entropy (stable)    (M6)
  viz.py           # draw_graph (uses _op)              (M3)
  seed.py          # single seed constant + seed_everything()
tests/
  test_gradcheck.py
  test_ops.py
examples/
  train_moons.py
  train_mnist.py
benchmarks/
  pytorch_baseline.py
assets/            # committed figures (graph render, decision boundary, results)
requirements.txt        # core deps (M0–M6), pinned
requirements-bench.txt  # torch, for the M7 benchmark only
```

## License

TODO.
