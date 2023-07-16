import sympy as sp
from sympy.utilities.codegen import codegen, C99CodeGen
from sympy.codegen.rewriting import create_expand_pow_optimization, optimize, optims_c99

def density_fn(r):
    h1 = 20.0 / 960.0
    nc = 3e11 * sp.exp(-(r-1.0) / h1)
    return 4.8e9/r**14 + 3e8/r**6 + 1.39e6/r**(2.3) + nc

def omega_pe(r):
    return 8.98e3 * sp.sqrt(density_fn(r)) * 2.0 * sp.pi

def optim(expr):
    return optimize(create_expand_pow_optimization(16)(expr), optims_c99)

class InlineC99CodeGen(C99CodeGen):
    def _get_routine_opening(self, routine):
        return [f"inline {super()._get_routine_opening(routine)[0]}"]

    def _preprocessor_statements(self, prefix):
        code_lines = []
        code_lines.extend(self.preprocessor_statements)
        code_lines = ['{}\n'.format(l) for l in code_lines]
        return code_lines

class InlineCudaCodeGen(C99CodeGen):
    def _get_routine_opening(self, routine):
        return [f"__host__ __device__ inline {super()._get_routine_opening(routine)[0]}"]

    def _preprocessor_statements(self, prefix):
        code_lines = []
        code_lines.extend(self.preprocessor_statements)
        code_lines = ['{}\n'.format(l) for l in code_lines]
        return code_lines

file_opening = """#if !defined(DENSITY_MODEL_H)
#define DENSITY_MODEL_H
#ifdef __cplusplus
extern "C" {
#endif
"""
file_opening_cuda = "#include \"cuda_runtime.h\"\n" + file_opening
file_closing = """#ifdef __cplusplus
}
#endif
#else
#endif
"""

if __name__ == '__main__':
    r = sp.symbols('r')
    density_r = density_fn(r)
    omega_pe_r = omega_pe(r)
    domega_dr = sp.diff(omega_pe(r), r)

    gen = InlineC99CodeGen(cse=True)
    source, header = codegen([
        ('density_r', optim(density_r)),
        ('omega_pe', optim(omega_pe_r)),
        ('domega_dr', optim(domega_dr)),
        ], prefix="DensityModel", header=False, empty=False, code_gen=gen)

    filename = source[0].replace('.c', '.h')
    with open(filename, 'w') as f:
        f.write(file_opening)
        f.write(source[1])
        f.write(file_closing)

    gen = InlineCudaCodeGen(cse=True)
    source, header = codegen([
        ('density_r', optim(density_r)),
        ('omega_pe', optim(omega_pe_r)),
        ('domega_dr', optim(domega_dr)),
        ], prefix="DensityModelCuda", header=False, empty=False, code_gen=gen)

    filename = source[0].replace('.c', '.h')
    with open(filename, 'w') as f:
        f.write(file_opening_cuda)
        f.write(source[1])
        f.write(file_closing)