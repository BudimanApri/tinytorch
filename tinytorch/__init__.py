"""tinytorch — a scalar reverse-mode autodiff engine that grows into a tiny
neural-network library.

See ``CLAUDE.md`` for the full spec and milestone plan. The engine core
(``engine.py``) is implemented by hand, one gradient-checked operation at a time.
"""
from .seed import SEED, seed_everything

__all__ = ["SEED", "seed_everything"]
__version__ = "0.0.0"
