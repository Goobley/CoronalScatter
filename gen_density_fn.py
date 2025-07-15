import re

import numpy as np
import sympy as sp
from sympy.utilities.codegen import codegen, C99CodeGen
from sympy.codegen.rewriting import create_expand_pow_optimization, optimize, optims_c99

def density_fn(r):
    h1 = 20.0 / 960.0
    nc = 3e11 * sp.exp(-(r-1.0) / h1)
    return 4.8e9/r**14 + 3e8/r**6 + 1.39e6/r**(2.3) + nc

def omega_pe(r):
    return 8.98e3 * sp.sqrt(density_fn(r)) * 2.0 * np.pi

def nu_scat_krupar(r, omega, eps):
    l_i = r * 1e5
    l_0 = 0.23 * 6.9e10 * r**0.82

    w_pe = omega_pe(r)

    nu_s = np.pi * eps**2 / (l_i**(1.0/3.0) * l_0**(2.0/3.0))
    # NOTE(cmo): This seems to give the cleanest codegen, with the fewest
    # transcendental calls. Using integer 4 causes issues due to our ispc
    # reprocessing adding d as a suffix.
    nu_s *= w_pe**4.0 * 2.998e10 / omega / (omega*omega - w_pe*w_pe)**(1.5)
    # nu_s *= w_pe*w_pe*w_pe*w_pe * 2.998e10 / omega / (omega*omega - w_pe*w_pe)**(1.5)
    return nu_s

def optim(expr):
    return optimize(create_expand_pow_optimization(30)(expr), optims_c99)

class InlineKokkosCodeGen(C99CodeGen):
    def _get_routine_opening(self, routine):
        return [f"KOKKOS_INLINE_FUNCTION {super()._get_routine_opening(routine)[0]}"]

    def _preprocessor_statements(self, prefix):
        code_lines = []
        code_lines.append('#ifndef ISPC')
        code_lines.append('#include <math.h>')
        code_lines.append('#endif')
        code_lines.append('#ifndef M_PI')
        # code_lines.append('#ifndef ISPC')
        code_lines.append('#define M_PI 3.14159265358979323846')
        # code_lines.append('#else')
        # code_lines.append('#define M_PI 3.14159265358979323846d')
        # code_lines.append('#endif')
        code_lines.append('#endif')
        code_lines = ['{}\n'.format(l) for l in code_lines]
        return code_lines

file_opening = """#if !defined(DENSITY_MODEL_H)
#define DENSITY_MODEL_H
"""
file_opening_kokkos = "#include \"Constants.hpp\"\n" + file_opening
file_closing = """#else
#endif
"""

if __name__ == '__main__':
    r, omega, eps = sp.symbols('r,omega,eps')
    density_r = density_fn(r)
    omega_pe_r = omega_pe(r)
    domega_dr = sp.diff(omega_pe(r), r)
    nu_scat = nu_scat_krupar(r, omega, eps)

    gen = InlineKokkosCodeGen(cse=True)
    source, header = codegen([
        ('density_r', optim(density_r)),
        ('omega_pe', optim(omega_pe_r)),
        ('domega_dr', optim(domega_dr)),
        ], prefix="source/DensityModel", header=False, empty=False, code_gen=gen)

    filename = source[0].replace('.c', '.hpp')
    with open(filename, 'w') as f:
        f.write(file_opening_kokkos)
        f.write(source[1])
        f.write(source_nu[1])
        f.write("#else\n")
        f.write(re.sub("([1-9][0-9]*\.?[0-9]*([Ee][+-]?[0-9]+)?)", "\g<0>d", source[1]))
        f.write(re.sub("([1-9][0-9]*\.?[0-9]*([Ee][+-]?[0-9]+)?)", "\g<0>d", source_nu[1]))
        f.write("#endif\n")
        f.write(file_closing)