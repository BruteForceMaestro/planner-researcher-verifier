from sympy import (
    symbols,
    sympify,
    integrate,
    diff,
    limit,
    series,
    solve,
    Eq,
    lambdify,
    expand,
    factor,
    apart,
    together,
    collect,
    dsolve,
    Function,
    fourier_transform,
    inverse_fourier_transform,
    laplace_transform,
    inverse_laplace_transform,
)
import mpmath as mp

def _parse_expr(expr_str: str, var: str | None = None, extra_symbols: list[str] | None = None):
    """
    Parse a string expression, making sure the main variable and any provided
    parameter names are treated as real symbols. This avoids accidental clashes
    with SymPy builtins like the gamma function.
    """
    locals_: dict[str, object] = {}
    v = None
    if var is not None:
        v = symbols(var, real=True)
        locals_[var] = v
    for name in extra_symbols or []:
        locals_[name] = symbols(name, real=True)
    # Common damping parameter "gamma" shadows sympy.gamma; treat it as a symbol
    if "gamma" in expr_str and "gamma(" not in expr_str and "gamma " not in expr_str:
        locals_.setdefault("gamma", symbols("gamma", real=True))
    expr = sympify(expr_str, locals=locals_)
    return expr, v


def sympy_simplify(expr_str: str, assumptions: str = "") -> str:
    # For now we ignore `assumptions` (you can wire this into sympy.assumptions later)
    expr, _ = _parse_expr(expr_str)
    return str(expr.simplify())


def sympy_integrate(
    expr_str: str,
    var: str,
    lower: str | None = None,
    upper: str | None = None,
) -> str:
    expr, v = _parse_expr(expr_str, var=var)
    if lower is None or upper is None:
        res = integrate(expr, v)
    else:
        a, _ = _parse_expr(lower)
        b, _ = _parse_expr(upper)
        res = integrate(expr, (v, a, b))
    return str(res)


def sympy_diff(expr_str: str, var: str, order: int = 1) -> str:
    expr, v = _parse_expr(expr_str, var=var)
    res = diff(expr, (v, order))
    return str(res)


def sympy_limit(expr_str: str, var: str, point: str, direction: str = "+") -> str:
    expr, v = _parse_expr(expr_str, var=var)
    p, _ = _parse_expr(point)
    res = limit(expr, v, p, dir=direction)
    return str(res)


def sympy_series(
    expr_str: str,
    var: str,
    point: str = "0",
    order: int = 5,
) -> str:
    expr, v = _parse_expr(expr_str, var=var)
    p, _ = _parse_expr(point)
    res = series(expr, v, p, order)
    return str(res)


def sympy_solve_equation(eq_str: str, var: str) -> str:
    # eq_str like "sin(x) = x/2"
    # be robust to extra '=' signs: split once
    if "=" not in eq_str:
        raise ValueError("Equation must contain '='.")
    left_str, right_str = eq_str.split("=", 1)

    v = symbols(var, real=True)
    locals_ = {var: v}

    left = sympify(left_str, locals=locals_)
    right = sympify(right_str, locals=locals_)

    sol = solve(Eq(left, right), v)
    return str(sol)


def numeric_check_equality(
    expr1_str: str,
    expr2_str: str,
    var: str,
    points: list[float],
) -> dict:
    v = symbols(var, real=True)
    locals_ = {var: v}

    e1 = sympify(expr1_str, locals=locals_)
    e2 = sympify(expr2_str, locals=locals_)

    f1 = lambdify(v, e1, "mpmath")
    f2 = lambdify(v, e2, "mpmath")

    results: list[float | str] = []
    for p in points:
        xval = mp.mpf(p)
        try:
            v1 = f1(xval)
            v2 = f2(xval)
            results.append(float(v1 - v2))
        except Exception as e:
            results.append(str(e))

    return {"points": points, "diffs": results}

def sympy_expand(expr_str: str, var: str | None = None) -> str:
    """
    Expand a symbolic expression (distribute products, powers, etc.).
    """
    expr, _ = _parse_expr(expr_str, var=var)
    return str(expand(expr))


def sympy_factor(expr_str: str, var: str | None = None) -> str:
    """
    Factor a polynomial expression (over the rationals by default).
    """
    expr, _ = _parse_expr(expr_str, var=var)
    return str(factor(expr))


def sympy_apart(expr_str: str, var: str) -> str:
    """
    Partial fraction decomposition with respect to `var`.
    """
    expr, v = _parse_expr(expr_str, var=var)
    return str(apart(expr, v))


def sympy_together(expr_str: str, var: str | None = None) -> str:
    """
    Combine terms over a common denominator.
    """
    expr, _ = _parse_expr(expr_str, var=var)
    return str(together(expr))


def sympy_collect(expr_str: str, var: str) -> str:
    """
    Collect coefficients by powers of `var`.
    """
    expr, v = _parse_expr(expr_str, var=var)
    return str(collect(expr, v))

def sympy_solve_system(eq_list: list[str], var_list: list[str]) -> str:
    """
    Solve a system of equations in the given variables.

    eq_list: list of strings, each either 'expr' (interpreted as expr = 0)
             or 'lhs = rhs'.
    var_list: list of variable names, e.g. ["x", "y"].
    """
    # Create all variables as real symbols
    vars_syms = {name: symbols(name, real=True) for name in var_list}

    equations = []
    for eq_str in eq_list:
        if "=" in eq_str:
            left_str, right_str = eq_str.split("=", 1)
            left = sympify(left_str, locals=vars_syms)
            right = sympify(right_str, locals=vars_syms)
            equations.append(Eq(left, right))
        else:
            expr = sympify(eq_str, locals=vars_syms)
            equations.append(Eq(expr, 0))

    sols = solve(equations, [vars_syms[name] for name in var_list], dict=True)
    return str(sols)

def sympy_dsolve(ode_str: str, func_name: str, var: str) -> str:
    """
    Solve an ODE using SymPy's dsolve.

    - `ode_str` should be a string for an expression that equals 0,
      e.g. 'y(t).diff(t,2) + 2*gamma*y(t).diff(t) + omega_0**2*y(t) - exp(-a*t)*cos(b*t)'.
    - Derivatives MUST be written using .diff syntax, not primes, e.g. y(t).diff(t,2) not y''(t).
    - `func_name` is the dependent function name, e.g. 'y'.
    - `var` is the independent variable name, e.g. 't'.
    """
    try:
        t = symbols(var, real=True)
        f = Function(func_name)

        # Important: map func_name -> f (the function), NOT f(t)
        locals_ = {
            var: t,
            func_name: f,
        }

        ode_expr = sympify(ode_str, locals=locals_)
        # Interpret ode_expr = 0 as the equation
        ode_eq = Eq(ode_expr, 0)

        sol = dsolve(ode_eq, f(t))
        return str(sol)
    except Exception as e:
        return f"Error: {e}"

def sympy_fourier_transform(expr_str: str, var: str, omega: str) -> str:
    """
    Compute the (unitary) Fourier transform of expr_str in variable `var`
    with frequency variable `omega`.
    """
    expr, v = _parse_expr(expr_str, var=var)
    w = symbols(omega, real=True)
    F = fourier_transform(expr, v, w)
    return str(F)


def sympy_inverse_fourier_transform(expr_str: str, omega: str, var: str) -> str:
    """
    Compute the inverse Fourier transform of expr_str, from `omega` to `var`.
    """
    expr, w = _parse_expr(expr_str, var=omega)
    x = symbols(var, real=True)
    f = inverse_fourier_transform(expr, w, x)
    return str(f)


def sympy_laplace_transform(expr_str: str, var: str, s: str) -> str:
    """
    Compute the Laplace transform of expr_str in variable `var` with parameter `s`.
    """
    expr, t = _parse_expr(expr_str, var=var)
    s_sym = symbols(s, real=True)
    F, region, cond = laplace_transform(expr, t, s_sym)
    # We return just F; region/cond can be added later if you want.
    return str(F)


def sympy_inverse_laplace_transform(expr_str: str, s: str, var: str) -> str:
    """
    Compute the inverse Laplace transform of expr_str from `s` to `var`.
    """
    expr, s_sym = _parse_expr(expr_str, var=s)
    t = symbols(var, real=True)
    f = inverse_laplace_transform(expr, s_sym, t)
    return str(f)
