import meshio
import numpy as np
# import math
from helpers import getNormals, getAbsNormals, getNeighbours, extrudePoints, getNeighboursV3, getNeighboursV4, getVertexNormal
from sympy import * # symbols, Eq, solve, Matrix
# import sympy

# testP1 = np.array([[0.5, 0.5, 0.5], [0.1, 0.7, 0.4], [0.2, 1.1, 0.6], [0.6, 1.2, 0.7]])
# testP2 = np.array([[0.5, 0.5, 0.5], [0.6, 1.2, 0.7], [0.9, 0.9, 0.8], [1.1, 0.6, 0.6]])
# testP3 = np.array([[0.5, 0.5, 0.5], [1.1, 0.6, 0.6], [1.2, 0.2, 0.4], [0.7, 0.0, 0.3]])
# testP4 = np.array([[0.5, 0.5, 0.5], [0.7, 0.0, 0.3], [0.0, -0.1, 0.7], [0.1, 0.7, 0.4]])
# nP = getNormals2(testP4)
# print(nP)


# surfaceMesh = meshio.read("PWsurface.obj")
surfaceMesh = meshio.read("./testGeometries/sphere.obj")

##layer 1 thickness##
t = 0.1

##get cells##
triangle_cells = np.array([])
quad_cells = np.array([])
for cell in surfaceMesh.cells:
    if cell.type == "triangle":
        if triangle_cells.size == 0:
            triangle_cells = cell.data
        else:
            triangle_cells = np.concatenate((triangle_cells, cell.data))
        # print(cell.data)
    elif cell.type == "quad":
        if quad_cells.size == 0:
            quad_cells = cell.data
        else:
            quad_cells = np.concatenate((quad_cells, cell.data))

##get surface points##
surfacePoints = surfaceMesh.points

# nVertex = getVertexNormal(10, surfacePoints, triangle_cells, quad_cells)
# print("nVertex normalized", nVertex)

## build derivatives ##
## for M >= 5##
P1 = 10 # for P=10 n_neighbours = 5 (2xtri, 3xquad)
v = 0
for cell in triangle_cells:
    if P1 in cell:
        v+=1
for cell in quad_cells:
    if P1 in cell:
        v+=1
print("valence", v)
print(" ")
r_xi = np.zeros([3, 1])
r_xi_1 = np.zeros([3, v])
r_0 = surfacePoints[P1]
# print(r_xi)
print("P1", r_0)
# P1neighbours, P1diagonals = getNeighboursV4(P1, triangle_cells, quad_cells)
P1neighbours = getNeighbours(P1, triangle_cells, quad_cells)
print("neighbours", P1neighbours)
# print("diagonals", P1diagonals)
# print("xi", r_xi_1)
print("neighbour A", surfacePoints[8])
print("neighbour B", surfacePoints[9])
print("neighbour C", surfacePoints[11])
print("neighbour D", surfacePoints[220])
print("neighbour E ", surfacePoints[221])

# get x_xi
for i, P in enumerate(P1neighbours):
    r_xi_1[:, i] = surfacePoints[P]
print("r_xi_1", r_xi_1)

cos_theta_m = np.zeros([v, 1])
for i in range(v):
    cos_theta_m[i] = np.cos((i * 2 * np.pi) / v)
print("cos_theta_m", cos_theta_m, "sum: ", np.sum(cos_theta_m))

r_xi = (2 / v) * np.dot(r_xi_1, cos_theta_m)
print("r_xi: ", r_xi)

# get r_eta
r_eta = np.zeros([3, 1])
r_eta_1 = np.zeros([3, v])
for i, P in enumerate(P1neighbours):
    r_eta_1[:, i] = surfacePoints[P]
print("r_eta_1", r_eta_1)

sin_theta_m = np.zeros([v, 1])
for i in range(v):
    sin_theta_m[i] = np.sin((i * 2 * np.pi) / v)
print("sin_theta_m", sin_theta_m, "sum: ", np.sum(sin_theta_m))

r_eta = (2 / v) * np.dot(r_eta_1, sin_theta_m)
print("r_eta: ", r_eta)

x, y, z = symbols('x y z')
# get r_xi_xi for M>=5
if v > 4:

    print(" ")

    r_xi_xi_trig_term = 4 * np.square(cos_theta_m) - 1
    print("r_xi_xi_trig_term: ", r_xi_xi_trig_term, "sum: ", np.sum(r_xi_xi_trig_term))

    r_xi_xi_1 = np.dot(r_xi_1, r_xi_xi_trig_term) #np.zeros([3, 1])
    print("r_xi_xi_1: ", r_xi_xi_1)

    # x, y, z = symbols('x y z')
    r_xi_xi_2 = Matrix([x, y, z])
    print("r_xi_xi_2: ", r_xi_xi_2, "shape: ", r_xi_xi_2.shape)
    r_xi_xi_2 = np.sum(r_xi_xi_trig_term) * r_xi_xi_2
    print("r_xi_xi_2*trig: ", r_xi_xi_2)
    r_xi_xi = (2 / v) * Matrix([r_xi_xi_1 - r_xi_xi_2])
    print("r_xi_xi: ", r_xi_xi)


# get r_xi_xi for M=4
else:
    print(" ")
    print(r_xi_1[:, 0])
    r_xi_xi_1 = r_xi_1[:, 0] + r_xi_1[:, 2]
    print(" ")
    print(r_xi_xi_1)
    # x, y, z = symbols('x y z')
    r_xi_xi = Matrix([r_xi_xi_1[0] - 2 * x, r_xi_xi_1[1] - 2 * y, r_xi_xi_1[2] - 2 * z])
    # r_xi_xi = r_xi_xi_1 - r_xi_xi_2
    print(" ")
    print("r_xi_xi", r_xi_xi)

# get r_eta_eta for M>=5
if v > 4:

    print(" ")

    r_eta_eta_trig_term = 4 * np.square(sin_theta_m) - 1
    print("r_eta_eta_trig_term: ", r_eta_eta_trig_term, "sum: ", np.sum(r_eta_eta_trig_term))

    r_eta_eta_1 = np.dot(r_eta_1, r_eta_eta_trig_term) #np.zeros([3, 1])
    print("r_eta_eta_1: ", r_eta_eta_1)

    # x, y, z = symbols('x y z')
    r_eta_eta_2 = Matrix([x, y, z])
    print("r_eta_eta_2: ", r_eta_eta_2, "shape: ", r_eta_eta_2.shape)
    r_eta_eta_2 = np.sum(r_eta_eta_trig_term) * r_eta_eta_2
    print("r_eta_eta_2*trig: ", r_eta_eta_2)
    r_eta_eta = (2 / v) * Matrix([r_eta_eta_1 - r_eta_eta_2])
    print("r_eta_eta: ", r_eta_eta)

# get r_eta_eta for M=4
else:
    print(" ")
    print(r_eta_1[:, 1])
    r_eta_eta_1 = r_eta_1[:, 1] + r_xi_1[:, 3]
    print(" ")
    print(r_eta_eta_1)
    # x, y, z = symbols('x y z')
    r_eta_eta = Matrix([r_eta_eta_1[0] - 2 * x, r_eta_eta_1[1] - 2 * y, r_eta_eta_1[2] - 2 * z])
    # r_xi_xi = r_xi_xi_1 - r_xi_xi_2
    print(" ")
    print("r_eta_eta", r_eta_eta)


# get r_xi_eta for M>=5
if v > 4:

    print(" ")
    xi_eta_trig_term = cos_theta_m * sin_theta_m
    print("cos_theta_m", cos_theta_m)
    print("sin_theta_m", sin_theta_m)
    print("xi_eta_trig_term", xi_eta_trig_term)
    print("r_xi_1: ", r_xi_1)
    r_xi_eta = (8 / v) * np.dot(r_xi_1, xi_eta_trig_term)
    print("r_xi_eta: ", r_xi_eta)



# # get r_xi_xi for M=4
# else:
#     print(" ")
#
#     r_xi_xi_trig_term = np.square(cos_theta_m)
#     print("r_xi_xi_trig_term: ", r_xi_xi_trig_term, "sum: ", np.sum(r_xi_xi_trig_term))
#
#     r_xi_xi_1 = np.dot(r_xi_1, r_xi_xi_trig_term) #np.zeros([3, 1])
#     print("r_xi_xi_1: ", r_xi_xi_1)
#
#     x, y, z = symbols('x y z')
#     r_xi_xi_2 = Matrix([x, y, z])
#     print("r_xi_xi_2: ", r_xi_xi_2, "shape: ", r_xi_xi_2.shape)
#     r_xi_xi_2 = np.sum(r_xi_xi_trig_term) * r_xi_xi_2
#     print("r_xi_xi_2*trig: ", r_xi_xi_2)
#     r_xi_xi = (2 / v) * Matrix([r_xi_xi_1 - r_xi_xi_2])
#     print("r_xi_xi: ", r_xi_xi)


# r_xi_1 = r_xi_1 - np.vstack(r_0)
# print("xi-r", r_xi_1)
# theta_m = np.zeros([v, 1])
# print("theta_m ini", theta_m)
# for i in range(v):
    # theta_m[i] = np.cos((i * 2 * np.pi) / v)
# print("theta_m", theta_m)


#
# M = Matrix(r_xi_1)
# x, y, z = symbols('x y z')
# r = Matrix([[x], [y], [z]])
# # print(M[:, 1])
# for i in range(v):
#     M[:, i] -= r
# print(M)
# # cosTheta_m = np.zeros([v, 1])
# # for i in range(v):
# #     theta_m
# #     cosTheta_m[i] = np.cos()
# theta_m = Matrix(theta_m)
# print(theta_m)
# # f_xi = (2 / v) * M * theta_m
# f_xi = M * theta_m
# print(f_xi)
# # cosTheta_m = Matrix( symarray('b', (3,4)) )
#


#
# ##get layer 1 points##
# l1_points = extrudePoints(surfacePoints, triangle_cells, quad_cells, t)
#
# ##assemble voxel points##
# l1_voxelPoints = np.concatenate((surfacePoints, l1_points))
#
# ##assemble layer 1 cells##
# nSurfacePoints = len(surfacePoints)
# l1_triangle_cells = triangle_cells + nSurfacePoints
# l1_quad_cells = quad_cells + nSurfacePoints
#
# ##assemble voxels##
# l1_voxels = [["wedge", np.concatenate((triangle_cells, l1_triangle_cells),axis=1)], ["hexahedron", np.concatenate((quad_cells, l1_quad_cells),axis=1)]]
#
# ##export mesh##
# l1_mesh = meshio.Mesh(points = l1_voxelPoints, cells = l1_voxels)
# meshio.write("./output/normalExtrudeAll2.vtk", l1_mesh, file_format="vtk", binary=False)
#
