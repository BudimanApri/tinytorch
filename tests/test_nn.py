import pytest
from tinytorch.engine import Value
from tinytorch.nn import Neuron, Layer, MLP

def test_neuron():
    n = Neuron(3)
    x = [Value(2.0), Value(3.0), Value(-1.0)]
    y = n(x)
    assert isinstance(y, Value)
    assert len(n.parameters()) == 4
    
    # check backward path
    y.backward()
    for p in n.parameters():
        assert p.grad != 0.0

def test_layer():
    l = Layer(3, 2)
    x = [Value(2.0), Value(3.0), Value(-1.0)]
    y = l(x)
    assert isinstance(y, list) and len(y) == 2
    assert len(l.parameters()) == 8
    
    loss = y[0] + y[1]
    loss.backward()
    for p in l.parameters():
        assert p.grad != 0.0

def test_mlp():
    mlp = MLP(3, [4, 4, 1])
    x = [Value(2.0), Value(3.0), Value(-1.0)]
    y = mlp(x)
    assert isinstance(y, Value)  # output size 1 should return single Value
    # parameters:
    # layer 1: 3*4 weights + 4 biases = 16
    # layer 2: 4*4 weights + 4 biases = 20
    # layer 3: 4*1 weights + 1 bias = 5
    # Total = 41
    assert len(mlp.parameters()) == 41
    
    y.backward()
    for p in mlp.parameters():
        assert p.grad != 0.0
