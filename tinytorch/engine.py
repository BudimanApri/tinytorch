"""tinytorch.engine — reverse-mode automatic differentiation core.

This is the heart of the project and is implemented BY HAND by the builder, one
operation at a time, each verified with a finite-difference gradient check
before it counts as done (see ``tests/test_gradcheck.py`` and ``CLAUDE.md``):

    M1 — ``Value`` + ``__add__``, ``__mul__``, ``backward()``
    M2 — ``exp``, ``tanh``, ``relu``, ``__pow__`` and the composed / right-side dunders
    M5 — ``Tensor`` refactor (NumPy-backed) with broadcasting-aware backward
"""
import math
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
    
    def tanh(self):
        # TODO: Implement tanh
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t**2)*out.grad
        out._backward = _backward
        return out

    def exp(self):
        x = math.exp(self.data)
        out = Value(x, (self,), 'exp')
        def _backward():
            self.grad += out.data*out.grad
        out._backward = _backward
        return out

    def relu(self):
        out_data = max(0, self.data)
        out = Value(out_data, (self,), 'relu')
        def _backward():
            self.grad += (1.0 if out.data>0 else 0.0)*out.grad
        out._backward = _backward
        return out

    def __pow__(self, power):
        assert isinstance(power, (int, float)), "Power must be an integer or float"
        out_data = self.data ** power
        out = Value(out_data, (self,), f'**{power}')
        def _backward():
            self.grad += (power * self.data**(power-1)) * out.grad
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
    def __neg__(self):
        return self* -1.0
    def __sub__(self,other):
        return self + (-other)
    def __radd__(self, other):
        return self + other
    def __rsub__(self, other):
        return other + (-self)
    def __rmul__(self, other):
        return self * other
    def __truediv__(self, other):
        return self * other**-1
    def __rtruediv__(self, other):
        return other * self**-1
