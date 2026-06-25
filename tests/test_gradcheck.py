"""Finite-difference gradient checks — the project's correctness oracle.

For every op and every parameter, compare the analytic gradient (from
``backward()``) against a central finite difference
``(f(x + eps) - f(x - eps)) / (2 * eps)`` with ``eps ~= 1e-5`` and assert the
relative error is below tolerance. Populated as each op lands (M1 onward).

No "looks right" — this independent check is why the project is low-risk.
"""
