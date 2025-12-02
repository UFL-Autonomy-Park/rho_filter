import sympy as sp
import numpy as np

k1, k2, k3 = sp.symbols('k1 k2 k3', real=True, positive=True)

# Terms defined by inspection of rows 2, 3, 4 and cols 1, 3, 4
# The top row [0, I, 0, 0] simply permutes the system; we analyze the sub-block.

# Term definitions
term_21 = -(3 - k1**2 + (2*k1 + k2 + k3)*(k1 + k2))
term_23 = -((2*k1 + k2 + k3)*(k1 + k2) + 1)
term_24 = -(2*k1 + k2 + k3)

term_31 = -(k1 + k2)
term_33 = -(2*k1 + k2)
term_34 = -1

term_41 = -(1 + (k1 + k2)*(k3 - k1))
term_43 = -(k1 + k2)*(k3 - k1)
term_44 = -k3

M_reduced = sp.Matrix([
    [term_21, term_23, term_24],
    [term_31, term_33, term_34],
    [term_41, term_43, term_44]
])

det = sp.simplify(M_reduced.det())

# Define M_reduced symbolically as before, then:
print(f"Determinant Expression: {sp.simplify(M_reduced.det())}")

# Convert to numpy for numerical condition number
func_M = sp.lambdify((k1, k2, k3), M_reduced, 'numpy')

# Test Case: k=1 (or your specific operating point)
M_num = func_M(1.0, 1.0, 1.0) 
cond_num = np.linalg.cond(M_num)

print(f"Numerical Condition Number (k=1): {cond_num}")