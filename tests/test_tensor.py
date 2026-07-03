import pytest
import numpy as np
from tinytorch.engine import Tensor

def gradcheck_tensor(f, inputs, eps=1e-5, tol=1e-5):
    """
    Harness to verify analytical gradients of Tensor operations using central finite difference.
    f should take inputs and return a single scalar Tensor (shape () or sum-reduced).
    """
    # 1. Forward pass to get scalar output
    out = f(inputs)
    assert out.shape == () or out.shape == (1,), "gradcheck_tensor requires f to return a scalar"
    
    # 2. Backward pass
    # Zero gradients first
    def zero_grad(node, visited=None):
        if visited is None:
            visited = set()
        if node not in visited:
            visited.add(node)
            node.grad = np.zeros_like(node.data)
            for child in node._prev:
                zero_grad(child, visited)
    zero_grad(out)
    
    out.backward()
    
    # Save analytical gradients
    grads_analytical = [x.grad.copy() for x in inputs]
    
    # 3. Numerical gradients via central finite difference
    for i, x in enumerate(inputs):
        grad_numerical = np.zeros_like(x.data)
        
        # Iterate over all indices in the NumPy array
        it = np.nditer(x.data, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            orig_val = x.data[idx]
            
            # f(x + eps)
            x.data[idx] = orig_val + eps
            val_plus = float(f(inputs).data.sum())
            
            # f(x - eps)
            x.data[idx] = orig_val - eps
            val_minus = float(f(inputs).data.sum())
            
            # Restore
            x.data[idx] = orig_val
            
            # Compute numerical gradient
            grad_numerical[idx] = (val_plus - val_minus) / (2 * eps)
            it.iternext()
            
        # 4. Compare analytical and numerical
        denom = np.maximum(1e-9, np.abs(grads_analytical[i]) + np.abs(grad_numerical))
        rel_err = np.abs(grads_analytical[i] - grad_numerical) / denom
        max_err = rel_err.max()
        assert max_err < tol, f"Gradcheck failed for input {i}: max_err={max_err}, analytical=\n{grads_analytical[i]}\nnumerical=\n{grad_numerical}"

def test_tensor_add():
    # Simple addition
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])
    gradcheck_tensor(lambda inputs: (inputs[0] + inputs[1]).sum(), [a, b])

def test_tensor_mul():
    # Simple multiplication
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])
    gradcheck_tensor(lambda inputs: (inputs[0] * inputs[1]).sum(), [a, b])

def test_tensor_matmul():
    # Matrix multiplication
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0, 6.0], [7.0, 8.0]])
    gradcheck_tensor(lambda inputs: (inputs[0] @ inputs[1]).sum(), [a, b])

def test_tensor_pow():
    a = Tensor([1.5, 2.5, 3.5])
    gradcheck_tensor(lambda inputs: (inputs[0] ** 3).sum(), [a])

def test_tensor_exp():
    a = Tensor([-0.5, 0.5, 1.0])
    gradcheck_tensor(lambda inputs: inputs[0].exp().sum(), [a])

def test_tensor_tanh():
    a = Tensor([-0.5, 0.5, 1.0])
    gradcheck_tensor(lambda inputs: inputs[0].tanh().sum(), [a])

def test_tensor_relu():
    a = Tensor([-1.5, 0.5, 1.5])
    # Avoid testing exactly at 0.0 where derivative is non-differentiable
    gradcheck_tensor(lambda inputs: inputs[0].relu().sum(), [a])

def test_tensor_sum():
    # Sum reduction along specific axes
    a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    
    # 1. Sum all
    gradcheck_tensor(lambda inputs: inputs[0].sum(), [a])
    # 2. Sum along axis 0
    gradcheck_tensor(lambda inputs: inputs[0].sum(axis=0).sum(), [a])
    # 3. Sum along axis 1
    gradcheck_tensor(lambda inputs: inputs[0].sum(axis=1).sum(), [a])
    # 4. Sum along axis 0 keepdims=True
    gradcheck_tensor(lambda inputs: inputs[0].sum(axis=0, keepdims=True).sum(), [a])

def test_tensor_broadcasting_add():
    # Adding a (10,) tensor to a (3, 10) tensor
    a = Tensor(np.random.randn(3, 10))
    b = Tensor(np.random.randn(10))
    gradcheck_tensor(lambda inputs: (inputs[0] + inputs[1]).sum(), [a, b])

    # Adding a (1, 10) tensor to a (3, 10) tensor
    c = Tensor(np.random.randn(3, 10))
    d = Tensor(np.random.randn(1, 10))
    gradcheck_tensor(lambda inputs: (inputs[0] + inputs[1]).sum(), [c, d])

def test_tensor_broadcasting_mul():
    # Multiplying a (1, 10) tensor and a (3, 10) tensor
    a = Tensor(np.random.randn(3, 10))
    b = Tensor(np.random.randn(1, 10))
    gradcheck_tensor(lambda inputs: (inputs[0] * inputs[1]).sum(), [a, b])
