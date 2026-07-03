import pytest
import numpy as np
from tinytorch.engine import Tensor
from tinytorch.nn import Linear, Sequential, ReLU, Tanh
from tinytorch.optim import SGD, Adam
from tinytorch.functional import softmax, cross_entropy
from tests.test_tensor import gradcheck_tensor

def test_linear():
    model = Linear(3, 2)
    x = Tensor(np.random.randn(5, 3))
    y = model(x)
    assert y.shape == (5, 2)
    assert len(model.parameters()) == 2  # w and b
    
    # check backward pass
    y.sum().backward()
    assert model.w.grad is not None
    assert model.b.grad is not None

def test_sequential():
    model = Sequential(
        Linear(3, 4),
        ReLU(),
        Linear(4, 2),
        Tanh()
    )
    x = Tensor(np.random.randn(5, 3))
    y = model(x)
    assert y.shape == (5, 2)
    assert len(model.parameters()) == 4  # (w1, b1, w2, b2)

def test_sgd():
    w = Tensor([2.0, 3.0])
    w.grad = np.array([0.5, -0.5])
    optim = SGD([w], lr=0.1)
    optim.step()
    assert np.allclose(w.data, [1.95, 3.05])
    optim.zero_grad()
    assert np.all(w.grad == 0.0)

def test_adam():
    w = Tensor([2.0, 3.0])
    w.grad = np.array([0.5, -0.5])
    optim = Adam([w], lr=0.1)
    optim.step()
    # verify parameter moved in the opposite direction of gradient
    assert w.data[0] < 2.0
    assert w.data[1] > 3.0
    optim.zero_grad()
    assert np.all(w.grad == 0.0)

def test_softmax_grad():
    x = Tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
    gradcheck_tensor(lambda inputs: softmax(inputs[0]).sum(), [x])

def test_cross_entropy_grad():
    logits = Tensor([[1.0, 2.0, 0.0], [-1.0, 5.0, 2.0]])
    targets = Tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    gradcheck_tensor(lambda inputs: cross_entropy(inputs[0], inputs[1]), [logits, targets])
