import numpy as np

@np.vectorize
def inverse_solver(f, y, a, b, tol=1e-5):
    """
    return x s.t. f(x) = y
    this solver assume that f is monotonically increasing inside [a, b] and that x is in [a, b].
    It will do dichotomic search until it find x s.t. |f(x) - y| <= tol
    """
    x = (a+b)/2
    _y = f(x)
    while tol < np.abs(_y - y):
        if _y < y: a = x
        else: b = x
        x = (a+b)/2
        _y = f(x)
    return x
