import pytest
from tinytorch.engine import Value

def test_milestone1_example1():
    # Test 1 exercises +, *, and accumulation across two paths
    a = Value(2.0)
    b = Value(3.0)
    c = a * b          # 6
    d = a + b          # 5
    L = c * d          # 30
    L.backward()
    
    # Check forward values
    assert c.data == 6.0
    assert d.data == 5.0
    assert L.data == 30.0
    
    # Check backward gradients
    assert L.grad == 1.0
    assert c.grad == 5.0
    assert d.grad == 6.0
    assert a.grad == 21.0
    assert b.grad == 16.0

def test_milestone1_example2():
    # Test 2 exercises the self-is-other case (e*e) and the accumulation trap
    a = Value(-2.0)
    b = Value(3.0)
    c = a * b          # -6
    e = c + b          # -3
    d = e * e          # 9
    d.backward()
    
    # Check forward values
    assert c.data == -6.0
    assert e.data == -3.0
    assert d.data == 9.0
    
    # Check backward gradients
    assert d.grad == 1.0
    assert e.grad == -6.0
    assert c.grad == -6.0
    assert b.grad == 6.0
    assert a.grad == -18.0

def test_composed_dunders():
    x = Value(2.0)
    y = Value(3.0)
    
    # subtraction
    assert (x - y).data == -1.0
    assert (x - 2.0).data == 0.0
    assert (2.0 - x).data == 0.0
    
    # division
    assert (x / y).data == 2.0 / 3.0
    assert (x / 2.0).data == 1.0
    assert (2.0 / x).data == 1.0
    
    # right-side add and mul
    assert (2.0 + x).data == 4.0
    assert (2.0 * x).data == 4.0

def test_viz():
    from tinytorch.viz import draw_graph
    x = Value(2.0)
    y = Value(3.0)
    z = x * y + y
    dot = draw_graph(z)
    assert dot is not None


