import meshio
import numpy as np
from helpers import *

# x = np.eye(5)
# # y = np.arange(5)
# y = np.ones(5, dtype = object)
# print(y)
# z = 2 * y
# print(z)
# x[:, z:-(y)] += 10
# print(x)

x = np.array([[5, 4, 3], [6, 7, 8]])
y = np.array([[0, 1, 2], [3, 4, 5]])
print(x)
print(y)
print(x-y)
