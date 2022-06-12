from xml import dom
import sympy as sp
from sympy.utilities.codegen import codegen
from sympy.codegen.rewriting import create_expand_pow_optimization, optimize, optims_c99

def density_fn(r):
    h1 = 20.0 / 960.0
    nc = 3e11 * sp.exp(-(r-1.0) / h1)
    return 4.8e9/r**14 + 3e8/r**6 + 1.39e6/r**(2.3) + nc

def omega_pe(r):
    return 8.98e3 * sp.sqrt(density_fn(r)) * 2.0 * sp.pi

def zero(r):
    return r - r

def omega_pe_dr(r):
    expr = omega_pe(r)
    dexpr = sp.diff(expr, r)
    return sp.Array([expr, dexpr])


def optim(expr):
    return optimize(create_expand_pow_optimization(16)(expr), optims_c99)
if __name__ == '__main__':
    r = sp.symbols('r')
    density_r = density_fn(r)
    omega_pe_r = omega_pe(r)
    domega_dr = sp.diff(omega_pe(r), r)
    omega_pe_dr_expr = omega_pe_dr(r)
    source, header = codegen([
        ('density_r', optim(density_r)),
        ('omega_pe', optim(omega_pe_r)),
        ('domega_dr', optim(domega_dr)),
        ('omega_pe_dr', omega_pe_dr_expr),
        ], "C99", "density_model", header=False, empty=False)
    with open(source[0], 'w') as f:
        f.write('\n'.join(source[1].split('\n')[1:]))