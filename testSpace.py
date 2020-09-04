import meshio
import numpy as np
from helpers import *
from sympy import *

# x = np.eye(5)
# # y = np.arange(5)
# y = np.ones(5, dtype = object)
# print(y)
# z = 2 * y
# print(z)
# x[:, z:-(y)] += 10
# print(x)
x, y = symbols('x y')
eq1 = Eq(2*x - y - 1, 0) # 2x -y -1 = 0
eq2 = Eq(x + y - 5, 0)
sol = solve((eq1, eq2),(x, y))
print(eq1)
# print(sol.keys())
# print(sol.values())
# print(sol.items())
# # sol2 = np.array(sol.values())
# sol2 = np.array(list(sol.values()))
# print(sol2)
# print(sol2.shape)
