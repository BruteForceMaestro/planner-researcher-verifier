from sympy import symbols, sympify, integrate, diff, limit, series, solve, Eq
from sympy import lambdify
import mpmath as mp

def sympy_simplify(expr_str: str, assumptions: str = "") -> str:
    x = symbols('x', real=True)  # extend as needed
    expr = sympify(expr_str)
    return str(expr.simplify())

def sympy_integrate(expr_str: str, var: str, lower: str = None, upper: str = None) -> str:
    v = symbols(var, real=True)
    expr = sympify(expr_str)
    if lower is None or upper is None:
        res = integrate(expr, v)
    else:
        a = sympify(lower)
        b = sympify(upper)
        res = integrate(expr, (v, a, b))
    return str(res)

def sympy_diff(expr_str: str, var: str, order: int = 1) -> str:
    v = symbols(var, real=True)
    expr = sympify(expr_str)
    return str(diff(expr, (v, order)))

def sympy_limit(expr_str: str, var: str, point: str, direction: str = "+") -> str:
    v = symbols(var, real=True)
    expr = sympify(expr_str)
    p = sympify(point)
    return str(limit(expr, v, p, dir=direction))

def sympy_series(expr_str: str, var: str, point: str = "0", order: int = 5) -> str:
    v = symbols(var, real=True)
    expr = sympify(expr_str)
    p = sympify(point)
    return str(series(expr, v, p, order))

def sympy_solve_equation(eq_str: str, var: str) -> str:
    v = symbols(var, real=True)
    # eq_str like "sin(x) = x/2"
    left_str, right_str = eq_str.split("=")
    left = sympify(left_str)
    right = sympify(right_str)
    sol = solve(Eq(left, right), v)
    return str(sol)

def numeric_check_equality(expr1_str: str, expr2_str: str, var: str, points: list[float]) -> dict:
    v = symbols(var, real=True)
    e1 = sympify(expr1_str)
    e2 = sympify(expr2_str)
    f1 = lambdify(v, e1, "mpmath")
    f2 = lambdify(v, e2, "mpmath")
    results = []
    for p in points:
        xval = mp.mpf(p)
        try:
            v1 = f1(xval)
            v2 = f2(xval)
            results.append(float(v1 - v2))
        except Exception as e:
            results.append(str(e))
    return {"points": points, "diffs": results}
