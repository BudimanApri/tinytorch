"""tinytorch.engine — reverse-mode automatic differentiation core.

This is the heart of the project and is implemented BY HAND by the builder, one
operation at a time, each verified with a finite-difference gradient check
before it counts as done (see ``tests/test_gradcheck.py`` and ``CLAUDE.md``):

    M1 — ``Value`` + ``__add__``, ``__mul__``, ``backward()``
    M2 — ``exp``, ``tanh``, ``relu``, ``__pow__`` and the composed / right-side dunders
    M5 — ``Tensor`` refactor (NumPy-backed) with broadcasting-aware backward
"""

class Value:
    """A single scalar value and its gradient."""

    def __init__(self, data, _children=(), _op=''):
        self.data = float(data)
        self.grad = 0.0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        # TODO: Implement __add__
        # 1. Compute out_data
        # 2. Build the output Value node wrapping the result and recording _prev and _op
        # 3. Define the _backward closure to propagate gradients back to self and other
        # 4. Return the output node
        out_data = self.data + other.data
        out = Value(out_data, (self, other), '+')
        def _backward():
            self.grad += 1.0*out.grad
            other.grad += 1.0*out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        # TODO: Implement __mul__
        # 1. Compute out_data
        # 2. Build the output Value node wrapping the result and recording _prev and _op
        # 3. Define the _backward closure to propagate gradients back to self and other
        # 4. Return the output node
        out_data = self.data * other.data
        out = Value(out_data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data *out.grad
        out._backward = _backward
        return out

    def backward(self):
        # TODO: Implement backward
        # 1. Build a topological ordering of the graph using DFS
        # 2. Seed self.grad = 1.0
        # 3. Call _backward() on all nodes in reversed topological order (from outputs to inputs)
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in topo[::-1]:
            node._backward()

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
