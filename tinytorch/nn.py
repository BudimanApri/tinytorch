import random
import numpy as np
from tinytorch.engine import Value, Tensor

class Module:
    """Base class for all neural network modules (both scalar and tensor)."""
    
    def zero_grad(self):
        """Resets all parameter gradients to 0.0."""
        for p in self.parameters():
            if hasattr(p.grad, 'fill'):
                p.grad.fill(0.0)
            else:
                p.grad = 0.0

    def parameters(self):
        """Returns a list of trainable parameters (Value or Tensor objects)."""
        params = []
        # Walk attributes to find parameters and submodules
        for name, val in self.__dict__.items():
            if isinstance(val, (Value, Tensor)):
                params.append(val)
            elif isinstance(val, Module):
                params.extend(val.parameters())
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, (Value, Tensor)):
                        params.append(item)
                    elif isinstance(item, Module):
                        params.extend(item.parameters())
        return params

class Neuron(Module):
    """A single neuron wrapping weights and a bias."""
    
    def __init__(self, nin, nonlin=True):
        # Initialize weights randomly between -1 and 1
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        # Initialize bias to 0
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        # TODO: Implement the forward pass for a single neuron.
        # 1. Compute w * x + b (using the zip of weights and inputs, and summing them).
        # 2. If self.nonlin is True, apply tanh() to the sum.
        # 3. Return the resulting Value node.
        r"""This is a function f(x) = b + \sum_{i=0}^N x_i * w_i"""
        activation = sum(w_i*x_i for w_i, x_i in zip(self.w, x)) + self.b
        return activation.tanh() if self.nonlin else activation

    def parameters(self):
        # TODO: Return list of parameters for this neuron (weights + bias)
        return self.w + [self.b]
    
    def __repr__(self):
        return f"Neuron(n_in={len(self.w)}, nonlin={self.nonlin})"

class Layer(Module):
    """A layer of neurons."""
    
    def __init__(self, nin, nout, **kwargs):
        self.nin = nin
        self.nout = nout
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        # TODO: Implement the forward pass for the layer.
        # 1. Call each neuron on the inputs x.
        # 2. Return a single Value if the output size is 1, otherwise return a list of Value outputs.
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs)==1 else outs

    def parameters(self):
        # TODO: Collect and return parameters from all neurons in this layer.
        return [p for n in self.neurons for p in n.parameters()]
    
    def __repr__(self):
        return f"Layer({self.nin}, {self.nout}, nonlin={self.neurons[0].nonlin})"
        
class MLP(Module):
    """A Multi-Layer Perceptron."""
    
    def __init__(self, nin, nouts):
        # nin: number of inputs
        # nouts: list of layer output sizes (e.g. [16, 16, 1])
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=(i!=len(nouts)-1)) for i in range(len(nouts))]

    def __call__(self, x):
        # TODO: Implement sequential forward pass.
        # Pass the input x sequentially through all layers and return the final layer's output.
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        # TODO: Collect and return parameters from all layers in this MLP.
        return [p for layer in self.layers for p in layer.parameters()]

class Linear(Module):
    """A linear/dense projection layer: out = x @ w + b"""
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        # TODO: Initialize weight tensor (glorot/xavier style is recommended: standard normal / sqrt(in_features))
        # TODO: Initialize bias tensor (zeros) if bias is True
        self.w = Tensor(np.random.randn(in_features, out_features) / np.sqrt(in_features))
        self.b = Tensor(np.zeros(out_features)) if bias else None
        #raise NotImplementedError("Implement Linear layer __init__")

    def __call__(self, x):
        # TODO: Implement Linear forward projection (x @ w + b)
        # Ensure broadcasting works correctly when adding bias
        if self.b is None:
            return x @ self.w
        else:
            return x @ self.w + self.b
        #raise NotImplementedError("Implement Linear layer __call__")

class Sequential(Module):
    """A container for chaining modules sequentially."""
    
    def __init__(self, *modules):
        super().__init__()
        # TODO: Store the modules in a list
        self.modules = list(modules)
        #raise NotImplementedError("Implement Sequential __init__")

    def __call__(self, x):
        # TODO: Forward inputs sequentially through all stored modules
        for module in self.modules:
            x = module(x)
        return x
        #raise NotImplementedError("Implement Sequential __call__")

class ReLU(Module):
    """ReLU activation module."""
    def __call__(self, x):
        # TODO: Apply ReLU activation to the Tensor x
        return x.relu()
        #raise NotImplementedError("Implement ReLU module")

class Tanh(Module):
    """Tanh activation module."""
    def __call__(self, x):
        # TODO: Apply Tanh activation to the Tensor x
        return x.tanh()
        #raise NotImplementedError("Implement Tanh module")

