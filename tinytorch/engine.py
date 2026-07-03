"""tinytorch.engine — reverse-mode automatic differentiation core.

This is the heart of the project and is implemented BY HAND by the builder, one
operation at a time, each verified with a finite-difference gradient check
before it counts as done (see ``tests/test_gradcheck.py`` and ``CLAUDE.md``):

    M1 — ``Value`` + ``__add__``, ``__mul__``, ``backward()``
    M2 — ``exp``, ``tanh``, ``relu``, ``__pow__`` and the composed / right-side dunders
    M5 — ``Tensor`` refactor (NumPy-backed) with broadcasting-aware backward
"""
import math
import numpy as np

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
        out_data = self.data + other.data
        out = Value(out_data, (self, other), '+')
        def _backward():
            self.grad += 1.0*out.grad
            other.grad += 1.0*out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out_data = self.data * other.data
        out = Value(out_data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data *out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
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
        # Recursive DFS: graph depth is bounded by Python's recursion limit (~1000).
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

def unbroadcast(grad, target_shape):
    """
    Sum-reduces grad to match target_shape to handle NumPy broadcasting in the backward pass.
    """
    if grad.shape == target_shape:
        return grad
    
    # 1. Sum along prepended dimensions (e.g. target_shape (10,), grad shape (32, 10))
    num_prepended = len(grad.shape) - len(target_shape)
    for _ in range(num_prepended):
        grad = grad.sum(axis=0)
        
    # 2. Sum along dimensions that were broadcasted from 1 to N (e.g. target_shape (1, 10), grad shape (32, 10))
    for axis, dim in enumerate(target_shape):
        if dim == 1:
            grad = grad.sum(axis=axis, keepdims=True)
            
    return grad

class Tensor:
    """A multi-dimensional array and its gradient."""

    def __init__(self, data, _children=(), _op=''):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out_data = self.data + other.data
        out = Tensor(data=out_data, _children= (self, other), _op= '+')
        def _backward():
            self.grad += unbroadcast(out.grad, self.shape)
            other.grad += unbroadcast(out.grad, other.shape)
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out_data = self.data * other.data
        out = Tensor(data=out_data, _children = (self, other), _op = '*')
        def _backward():
            self.grad += unbroadcast(out.grad * other.data, self.shape)
            other.grad += unbroadcast(out.grad * self.data, other.shape)
        out._backward = _backward
        return out

    def __pow__(self, power):
        assert isinstance(power, (int, float)), "Power must be an integer or float"
        out_data = self.data**power
        out = Tensor(data = out_data, _children = (self,), _op = f'**{power}')
        def _backward():
            self.grad += power * self.data**(power-1) * out.grad
        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out_data = self.data @ other.data
        out = Tensor(data = out_data, _children = (self,other), _op = '@')
        def _backward():
            self.grad += unbroadcast(out.grad @ other.data.T, self.shape)
            other.grad += unbroadcast(self.data.T @ out.grad, other.shape)
        out._backward = _backward
        return out

    def exp(self):
        out_data = np.exp(self.data)
        out = Tensor(data = out_data, _children= (self,), _op = 'exp')
        def _backward():
            self.grad += out.grad * out_data
        out._backward = _backward
        return out

    def tanh(self):
        out_data = np.tanh(self.data)
        out = Tensor(data = out_data, _children= (self,), _op = 'tanh')
        def _backward():
            self.grad += out.grad * (1 - out_data**2)
        out._backward = _backward
        return out

    def relu(self):
        out_data = np.maximum(0, self.data)
        out = Tensor(data = out_data, _children= (self,), _op = 'relu')
        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out
    def log(self):
        # Compute forward log
        out_data = np.log(self.data)
        out = Tensor(data=out_data, _children=(self,), _op='log')
        def _backward():
            self.grad += out.grad / self.data
        out._backward = _backward
        return out



    def sum(self, axis=None, keepdims=False):
        out_data = self.data.sum(axis = axis, keepdims = keepdims)
        out = Tensor(data = out_data, _children = (self,), _op = 'sum')
        # If keepdims dropped the reduced axes, restore them as size-1 dims so
        # out.grad broadcasts back to self.shape.
        def _backward():
            grad = out.grad
            if not keepdims and axis is not None:
                axes = [axis] if isinstance(axis, int) else list(axis)
                shape = list(self.shape)
                for ax in axes:
                    pos_ax = ax if ax >=0 else len(shape) + ax
                    shape[pos_ax] = 1
                grad = grad.reshape(shape)
            self.grad += np.broadcast_to(grad, self.shape)
        out._backward = _backward
        return out

    def backward(self):
        # Recursive DFS: graph depth is bounded by Python's recursion limit (~1000).
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        # Seed gradient with 1.0 (matching the shape of this tensor)
        self.grad = np.ones_like(self.data)
        
        # Call _backward in reverse topological order
        for node in topo[::-1]:
            node._backward()

    def __repr__(self):
        return f"Tensor(shape={self.shape}, data={self.data.tolist()}, grad={self.grad.tolist()})"

    # Composed and right-side dunder methods
    def __neg__(self):
        return self * -1.0
        
    def __sub__(self, other):
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

