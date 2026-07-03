import numpy as np
from tinytorch.engine import Tensor

def softmax(x):
    """
    Computes the numerically stable softmax of input Tensor x.
    
    Args:
        x (Tensor): Input tensor of shape (batch_size, num_classes) or (num_classes,).
        
    Returns:
        Tensor: Softmax probabilities.
    """
    # Subtracting the row max keeps the exponents from overflowing; the max is a
    # constant w.r.t. the gradient, so correctness is unaffected.
    C = np.max(x.data, axis=-1, keepdims=True)
    exps = (x - C).exp()
    return exps / exps.sum(axis=-1, keepdims=True)

def cross_entropy(logits, targets):
    """
    Computes the numerically stable cross-entropy loss between logits and one-hot encoded targets.
    
    Args:
        logits (Tensor): Predicted logits of shape (batch_size, num_classes).
        targets (Tensor): One-hot encoded target labels of shape (batch_size, num_classes).
        
    Returns:
        Tensor: Mean cross-entropy loss (scalar Tensor).
    """
    # log-sum-exp trick: factor out the row max C so exp() never overflows.
    batch_size = logits.shape[0]
    C = np.max(logits.data, axis=-1, keepdims=True)
    log_sum_exp = (logits - C).exp().sum(axis=-1, keepdims=True).log() + C
    target_logits = (logits * targets).sum(axis=-1, keepdims=True)
    batch_loss = log_sum_exp - target_logits
    return batch_loss.sum() * (1.0 / batch_size)
