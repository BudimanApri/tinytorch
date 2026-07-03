import numpy as np
from tinytorch.engine import Tensor

class SGD:
    """Stochastic Gradient Descent optimizer with momentum."""
    
    def __init__(self, params, lr=1e-3):
        self.params = params
        self.lr = lr
        
    def step(self):
        # TODO: Implement parameter update step
        # For each parameter p in self.params: p.data -= self.lr * p.grad
        for p in self.params:
            p.data -= self.lr * p.grad
        #raise NotImplementedError("Implement SGD step")

    def zero_grad(self):
        # TODO: Reset gradients of all parameters to 0.0
        # Hint: use p.grad.fill(0.0) since p.grad is a NumPy array.
        for p in self.params:
            p.grad.fill(0.0)
        #raise NotImplementedError("Implement SGD zero_grad")

class Adam:
    """Adam optimizer."""
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        # TODO: Initialize moment buffers: self.m and self.v (lists of numpy zero arrays, matching shapes of params)
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        #raise NotImplementedError("Implement Adam __init__")

    def step(self):
        # TODO: Implement Adam update step
        # 1. Increment self.t
        # 2. For each parameter p and its index i:
        #    a. self.m[i] = beta1 * self.m[i] + (1 - beta1) * p.grad
        #    b. self.v[i] = beta2 * self.v[i] + (1 - beta2) * (p.grad ** 2)
        #    c. m_hat = self.m[i] / (1 - beta1**t)
        #    d. v_hat = self.v[i] / (1 - beta2**t)
        #    e. p.data -= lr * m_hat / (sqrt(v_hat) + eps)
        self.t += 1
        for i, p in enumerate(self.params):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad ** 2)
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        #raise NotImplementedError("Implement Adam step")

    def zero_grad(self):
        # TODO: Reset gradients of all parameters to 0.0
        for p in self.params:
            p.grad.fill(0.0)
        #raise NotImplementedError("Implement Adam zero_grad")
