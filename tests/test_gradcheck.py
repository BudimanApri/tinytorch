import pytest
from tinytorch.engine import Value

def gradcheck(f, inputs, eps=1e-5, tol=1e-6):
    """
    Computes numerical gradients using central finite difference and compares
    them against the analytical gradients computed via backward().
    """
    # 1. Forward pass to get output Value
    out = f(inputs)
    
    # 2. Backward pass to get analytical gradients
    # First, zero all gradients in the graph just in case
    def zero_grad(node, visited=None):
        if visited is None:
            visited = set()
        if node not in visited:
            visited.add(node)
            node.grad = 0.0
            for child in node._prev:
                zero_grad(child, visited)
    zero_grad(out)
    
    # Run backward
    out.backward()
    grads_analytical = [v.grad for v in inputs]
    
    # 3. Central finite difference for numerical gradients
    grads_numerical = []
    for i, x in enumerate(inputs):
        orig_data = x.data
        
        # f(x + eps)
        x.data = orig_data + eps
        out_plus = f(inputs)
        val_plus = out_plus.data
        
        # f(x - eps)
        x.data = orig_data - eps
        out_minus = f(inputs)
        val_minus = out_minus.data
        
        # Restore data
        x.data = orig_data
        
        # Central difference
        grad_num = (val_plus - val_minus) / (2 * eps)
        grads_numerical.append(grad_num)
        
    # 4. Compare analytical vs numerical gradients
    for i, (ga, gn) in enumerate(zip(grads_analytical, grads_numerical)):
        denom = max(1e-9, abs(ga) + abs(gn))
        rel_err = abs(ga - gn) / denom
        assert rel_err < tol, f"Gradcheck failed for input {i}: analytical={ga}, numerical={gn}, rel_err={rel_err}"

def test_add_gradcheck():
    # f(a, b) = a + b
    a = Value(2.3)
    b = Value(-4.5)
    gradcheck(lambda inputs: inputs[0] + inputs[1], [a, b])

def test_mul_gradcheck():
    # f(a, b) = a * b
    a = Value(2.3)
    b = Value(-4.5)
    gradcheck(lambda inputs: inputs[0] * inputs[1], [a, b])

def test_composed_gradcheck_1():
    # L = (a*b)*(a+b)
    a = Value(2.0)
    b = Value(3.0)
    gradcheck(lambda inputs: (inputs[0] * inputs[1]) * (inputs[0] + inputs[1]), [a, b])

def test_composed_gradcheck_2():
    # c = a*b; e = c+b; d = e*e
    a = Value(-2.0)
    b = Value(3.0)
    def f(inputs):
        c = inputs[0] * inputs[1]
        e = c + inputs[1]
        return e * e
    gradcheck(f, [a, b])

def test_tanh_gradcheck():
    a = Value(0.5)
    gradcheck(lambda inputs: inputs[0].tanh(), [a])

def test_exp_gradcheck():
    a = Value(-1.5)
    gradcheck(lambda inputs: inputs[0].exp(), [a])

def test_relu_gradcheck():
    a = Value(1.5)
    gradcheck(lambda inputs: inputs[0].relu(), [a])
    b = Value(-1.5)
    gradcheck(lambda inputs: inputs[0].relu(), [b])

def test_pow_gradcheck():
    a = Value(2.5)
    gradcheck(lambda inputs: inputs[0] ** 3, [a])
