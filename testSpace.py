import meshio
import numpy as np
from helpers import *
from sympy import *

x = [0, 1, 2]
x_reduced = [0, 2]
triFace = [0, 2, 5]
print("x", x)
print("triFace", triFace)
# unwanted = {'item', 5}
unwanted = [0, 2]
item_list = [0, 2, 5]
item_list = [e for e in item_list if e not in unwanted]
triFaceReduced = [e for e in triFace if e not in x]
print(item_list)
print(triFaceReduced)
