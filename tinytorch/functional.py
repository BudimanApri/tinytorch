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
    # TODO: Implement numerically stable softmax
    # 1. C = np.max(x.data, axis=-1, keepdims=True)
    # 2. Subtract C from x (for numerical stability: prevents large exponent overflow)
    # 3. Compute exps = (x - C).exp()
    # 4. Return exps / exps.sum(axis=-1, keepdims=True)
    C = np.max(x.data, axis=-1, keepdims=True)
    exps = (x - C).exp()
    return exps / exps.sum(axis=-1, keepdims=True)
    #raise NotImplementedError("Implement softmax")

def cross_entropy(logits, targets):
    """
    Computes the numerically stable cross-entropy loss between logits and one-hot encoded targets.
    
    Args:
        logits (Tensor): Predicted logits of shape (batch_size, num_classes).
        targets (Tensor): One-hot encoded target labels of shape (batch_size, num_classes).
        
    Returns:
        Tensor: Mean cross-entropy loss (scalar Tensor).
    """
    # TODO: Implement numerically stable cross entropy using the log-sum-exp trick
    # 1. C = np.max(logits.data, axis=-1, keepdims=True)
    # 2. Compute log_sum_exp = (logits - C).exp().sum(axis=-1, keepdims=True).log() + C
    # 3. Compute target_logits = (logits * targets).sum(axis=-1, keepdims=True)
    # 4. Compute batch loss = log_sum_exp - target_logits
    # 5. Return the mean loss across the batch: loss.sum() * (1.0 / batch_size)
    batch_size = logits.shape[0]
    C = np.max(logits.data, axis=-1, keepdims=True)
    log_sum_exp = (logits - C).exp().sum(axis=-1, keepdims=True).log() + C
    target_logits = (logits * targets).sum(axis=-1, keepdims=True)
    batch_loss = log_sum_exp - target_logits
    return batch_loss.sum() * (1.0 / batch_size)
    #raise NotImplementedError("Implement cross_entropy")
