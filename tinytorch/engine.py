"""tinytorch.engine — reverse-mode automatic differentiation core.

This is the heart of the project and is implemented BY HAND by the builder, one
operation at a time, each verified with a finite-difference gradient check
before it counts as done (see ``tests/test_gradcheck.py`` and ``CLAUDE.md``):

    M1 — ``Value`` + ``__add__``, ``__mul__``, ``backward()``
    M2 — ``exp``, ``tanh``, ``relu``, ``__pow__`` and the composed / right-side dunders
    M5 — ``Tensor`` refactor (NumPy-backed) with broadcasting-aware backward

Intentionally left empty for now: the implementation *is* the lesson.
"""
