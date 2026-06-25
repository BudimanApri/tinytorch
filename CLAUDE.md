# tinytorch — Project A: scalar autograd engine → tiny neural-net library

## What this project is

A from-scratch reverse-mode automatic-differentiation engine (micrograd-style) that
grows into a small neural-network library, trains an MLP, and is benchmarked against
PyTorch. It is the foundational project in a research-oriented CS portfolio aimed at
graduate-school / ML-systems applications.

**The point is deep understanding, not a finished clone.** A generated micrograd copy is
near-worthless for this goal — the value is the builder's ability to explain and defend
every line to a research advisor. Optimize for *learning* and *reproducibility* over speed
of completion.

## Who you're working with

- Intermediate programmer; primary interests are AI/ML/deep learning and systems.
- Has already worked through the *concepts* of reverse-mode autodiff by hand:
  computational graphs, the chain rule as graph traversal, forward-record / backward-fire,
  gradient accumulation (`+=` over every use of a node), and reverse-topological ordering.
  Can compute gradients on a small graph manually and has spotted the reused-node
  accumulation trap unaided.
- Learning style: concept-first, then implementation. Learns by attempting exercises and
  being challenged — not by reading code that was written for them.

## Operating contract — how you (Claude Code) should behave

This is a learning project, so default to **mentor / pair-programmer**, not autocomplete.

1. For the **autodiff core** — the `Value` class, each operation's local-gradient rule
   inside its `_backward` closure, and the topo-sorted `backward()` — do **not** write the
   implementation outright. Ask the user to attempt it, then review: point to the exact
   local rule or the exact accumulation that's wrong, show the discrepancy, and explain the
   *why*. Confirm correctness with a gradient check before moving on.
2. You **may** write directly (just say so clearly, so authorship stays honest): repo
   scaffolding, test harnesses, plotting, data loaders, the graph-viz helper, the PyTorch
   baseline, CI config, and general hygiene. These are tooling, not the lesson.
3. Advance **one milestone at a time**. After each: run the tests, commit to git with a
   clear message, and pause for the user before starting the next milestone.
4. **Every new operation must pass a finite-difference gradient check before it counts as
   done.** This is the non-negotiable correctness oracle. No "looks right" — prove it.
5. When the user's gradient is wrong, say so plainly and show analytic-vs-numeric numbers.
   Never paper over a bug to be encouraging.
6. Prefer prose explanations over walls of code. When you do show code, keep it minimal and
   tie it back to the math.

## Engine spec (the target design)

A `Value` is a node in the computational graph and stores exactly what the backward pass
needs:

- `data` — the forward scalar value.
- `grad` — starts at `0.0`; **accumulates** during backward (each consumer does `+=`).
- `_backward` — a closure (default no-op) that pushes this node's gradient to its inputs.
- `_prev` — the set of input nodes that produced this one (the recorded graph edges).
- `_op` — a string label for the operation; **used only by `__repr__` and the graph
  visualizer, never by the gradient math.** It earns its place by making nodes inspectable.

Every operation follows the same three-step shape:
(1) compute `out.data`; (2) build `out` recording `_prev` and `_op`; (3) *define* (do not
call) a `_backward` closure that applies the local rule and `+=` into each input's `.grad`.
The closure is written during the forward pass but executed during backward — it reads
`out.grad` at call time, by which point downstream nodes have already accumulated into it.

`backward()`: build a topological order via DFS (append each node *after* its children),
seed `self.grad = 1.0`, then call `_backward()` on nodes in **reversed** topo order so every
consumer fires before its producers — which is what makes each `+=` land before a node is
read.

**Local-gradient rules** (the user implements these; review against this table):

| op | local gradient |
|----|----------------|
| `a + b` | `∂/∂a = 1`, `∂/∂b = 1` (pass-through) |
| `a * b` | `∂/∂a = b`, `∂/∂b = a` (switcher; cache the operands) |
| `a ** n` (n const) | `∂/∂a = n * a**(n-1)` |
| `exp(a)` | `∂/∂a = exp(a)` (i.e. `out.data` — it's its own derivative) |
| `tanh(a)` | `∂/∂a = 1 - tanh(a)**2` (i.e. `1 - out.data**2`) |
| `relu(a)` | `∂/∂a = 1 if a > 0 else 0` |

**Two correctness landmines to watch for in review:**
- Accumulation: a node used in multiple downstream ops must `+=`, never `=`. Overwriting
  silently drops all paths but the last.
- The `a * a` (i.e. `e * e`) case: when `self is other`, the closure runs both `+=` lines on
  the *same* object, so `grad` correctly accumulates twice (the factor of 2 falls out — do
  not special-case it).

**Composed ops come free** (no new backward rules): `__neg__ = self * -1`,
`__sub__ = self + (-other)`, `__truediv__ = self * other**-1`, plus `__radd__`/`__rmul__`
(and `__rsub__`/`__rtruediv__`) so `2 * x` and `2 - x` work.

## Milestones

- **M0 — Repo setup.** Init git, virtualenv, pinned deps, `.gitignore`, README skeleton,
  a single seed constant. (You may scaffold this.)
- **M1 — Scalar core.** `Value` + `__add__`, `__mul__`, `backward()`. User implements.
  Verify by hand on `L = (a*b)*(a+b)` and on `c=a*b; e=c+b; d=e*e` (both already worked out
  by hand in prior sessions), then gradient-check.
- **M2 — Op coverage.** `exp`, `tanh`, `relu`, `__pow__`; the right-side and composed
  dunders. Each new local rule implemented by the user and gradient-checked individually.
- **M3 — Inspectability.** `__repr__` using `_op`; `viz.draw_graph(root)` (graphviz
  `Digraph`) that labels operation nodes by `_op` and shows `data`/`grad` per node. Render
  the worked-example graph as a sanity image. (You may write the viz — it's tooling.)
- **M4 — Scalar nn.** `Neuron` / `Layer` / `MLP` on top of `Value`; train on the two-moons
  toy dataset; hand-rolled SGD; plot the decision boundary. User wires the loop and must be
  able to explain `zero_grad → forward → loss → backward → step`.
- **M5 — Tensor refactor.** A `Tensor` wrapping a NumPy array; reimplement ops with array
  math; **handle broadcasting in the backward pass** by sum-reducing the gradient back to
  the input's original shape. This is the systems lesson — gradient-check broadcasting cases
  explicitly (e.g. bias `(10,)` added to batch `(32, 10)`).
- **M6 — nn / optim / functional.** `Module` base, `Linear`, `Sequential`, activation
  modules; `SGD` and `Adam`; numerically stable `softmax` and `cross_entropy` (log-sum-exp).
- **M7 — MNIST + benchmark.** Train an MLP on MNIST, report accuracy; build an equivalent
  PyTorch baseline; compare accuracy and honestly note the speed gap (scalar/NumPy will be
  far slower — that gap *is* a teaching result, not a failure).

## Repo structure

```
tinytorch/
  engine.py        # Value / Tensor + autodiff core
  nn.py            # Module, Linear, activations
  optim.py         # SGD, Adam
  functional.py    # softmax, cross-entropy (numerically stable)
  viz.py           # draw_graph (uses _op)
tests/
  test_gradcheck.py
  test_ops.py
examples/
  train_moons.py
  train_mnist.py
benchmarks/
  pytorch_baseline.py
README.md
requirements.txt   # pinned; or pyproject.toml
```

## Correctness oracle: gradient checking

For every op and every parameter, compare the analytic gradient (from `backward()`) against
a central finite difference `(f(x + ε) − f(x − ε)) / (2ε)` with `ε ≈ 1e-5`, and assert the
relative error is below a tolerance (e.g. `1e-6`). This independent check is why the project
is low-risk: you are never guessing whether a gradient is right. Put it in
`tests/test_gradcheck.py` and run it after every new op.

## Reproducibility & git conventions

- Seed everything (Python `random`, NumPy, Torch) from one constant; deterministic data
  splits.
- Pin dependencies; record exact versions.
- One commit per milestone with a clear, imperative message (e.g. `Add tanh op + gradcheck`).
- Keep examples runnable end-to-end from a clean checkout.

## README-as-mini-paper

The README is a portfolio artifact, not a usage note. It should cover: what autodiff is and
why reverse mode wins for ML (one scalar loss, many parameters → one backward pass); the
design (computational graph, the three-step op pattern, accumulation, topo order) with a
rendered graph diagram from `viz.draw_graph`; results (two-moons decision boundary, MNIST
accuracy vs the PyTorch baseline); and exact reproduction instructions.

## After Project A

The tensor refactor makes this the launchpad for Project 2 (a Transformer from scratch →
small GPT). Keep the engine's API clean enough that a later `nn` layer could be built on it
without fighting the core.
