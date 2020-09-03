import meshio
import numpy as np
# import math
from helpers import *
from sympy import * # symbols, Eq, solve, Matrix


surface_triangles, surface_quads, surface_points = read_surface("./testGeometries/sphere.obj")
d1 = 0.1
d2 = 0.12
level1, level2, level3 = build_ref_mesh_points(surface_points, surface_triangles, surface_quads, d1, d2, 1)
nSurfacePoints = len(surface_points)
level1_triangles = surface_triangles
level1_quads = surface_quads
level2_triangles = level1_triangles + nSurfacePoints
level2_quads = level1_quads + nSurfacePoints
level3_triangles = level2_triangles + nSurfacePoints
level3_quads = level2_quads + nSurfacePoints
allPoints = np.concatenate((level1, level2, level3))

faceTable = np.load('./dataOutput/faceTable.npy', allow_pickle=True)
vertexTable = np.load('./dataOutput/vertexTable.npy', allow_pickle=True)
level2VertexTable = np.load('./dataOutput/level2VertexTable.npy', allow_pickle=True)
level2FaceTable = np.load('./dataOutput/level2FaceTable.npy', allow_pickle=True)

directNeighbours = np.load('./dataOutput/directNeighboursTable.npy', allow_pickle=True)
valence = np.load('./dataOutput/valenceTable.npy', allow_pickle=True)
diagonals = np.load('./dataOutput/diagonalsTable.npy', allow_pickle=True)
nDiagonals = np.load('./dataOutput/nDiagonalsTable.npy', allow_pickle=True)


#######################################

# build r_zeta
# r_zeta = (vertexTable[:, [4, 5, 6]] * d1 + level2VertexTable[:, [4, 5, 6]] * d2) / (d1 + d2)
# print("r_zeta", r_zeta[0:6, :])

######################################

#build r_zeta_zeta
# r_zeta_zeta = (level2VertexTable[:, [4, 5, 6]] - vertexTable[:, [4, 5, 6]]) / (d1 + d2)
# print("r_zeta_zeta", r_zeta_zeta[0:6, :])

#######################################

# build r_xi[nSurfacePoints, 3] | [nSurfacePoints, [x_xi, y_xi, z_xi]]
# print(nDiagonals[0:100])
# maxM = max(max(valence), max(nDiagonals) +4)
# r_xi_trig = np.ndarray((maxM, 1), dtype = object)
# print(valence+nDiagonals)
# print(valence.shape)
# print(nDiagonals.shape)
# print(nDiagonals)

r_xi = np.ndarray((nSurfacePoints, 3), dtype = object)
r_eta = np.ndarray((nSurfacePoints, 3), dtype = object)
r_xi_xi = np.ndarray((nSurfacePoints, 3), dtype = object)
r_eta_eta = np.ndarray((nSurfacePoints, 3), dtype = object)
x, y, z = symbols('x y z')
for vertex in range(nSurfacePoints):
    M = valence[vertex, 0]
    r_trig_term = np.arange(M)
    r_xi_trig_term = np.cos((r_trig_term * 2 * np.pi) / M)
    r_eta_trig_term = np.sin((r_trig_term * 2 * np.pi) / M)

    neighbourlist = directNeighbours[vertex, :]

    neighbourCoor = np.ndarray((3, M), dtype = object)
    for m in range(M):
        neighbourCoor[: , m] = level2[neighbourlist[m], :]

    r_xi[vertex, :] = (2 / M) * np.dot(neighbourCoor, r_xi_trig_term)
    r_eta[vertex, :] = (2 / M) * np.dot(neighbourCoor, r_eta_trig_term)

    if M > 4:
        # evaluate r_xi_xi
        r_xi_xi_trig_term = 4 * np.square(r_xi_trig_term) - 1
        r_xi_xi_1 = np.reshape(np.dot(neighbourCoor, r_xi_xi_trig_term), (3, 1))
        r_xi_xi[vertex, [[0], [1], [2]]] = (2 / M) * Matrix([r_xi_xi_1 - np.sum(r_xi_xi_trig_term) * Matrix([x, y, z])])

        # evaluate r_eta_eta
        r_eta_eta_trig_term = 4 * np.square(r_eta_trig_term) - 1
        r_eta_eta_1 = np.reshape(np.dot(neighbourCoor, r_eta_eta_trig_term), (3, 1))
        r_eta_eta[vertex, [[0], [1], [2]]] = (2 / M) * Matrix([r_eta_eta_1 - np.sum(r_eta_eta_trig_term) * Matrix([x, y, z])])

    elif M == 4:
        # evaluate r_xi_xi
        r_xi_xi_1 = np.reshape((neighbourCoor[:, 0] + neighbourCoor[:, 2]), (3, 1))
        # print(neighbourCoor)
        # print(r_xi_xiTemp1)
        r_xi_xi[vertex, [[0], [1], [2]]] = Matrix([r_xi_xi_1 - 2 * Matrix([x, y, z])])

        # evaluate r_eta_eta
        r_eta_eta_1 = np.reshape((neighbourCoor[:, 1] + neighbourCoor[:, 3]), (3, 1))
        r_eta_eta[vertex, [[0], [1], [2]]] = Matrix([r_eta_eta_1 - 2 * Matrix([x, y, z])])

# print(r_xi)
# print(r_eta)
# print(r_xi_xi)
# print(r_eta_eta)
# print(valence[0:20])

#########################################

# # build r_xi_eta
r_xi_eta = np.ndarray((nSurfacePoints, 3), dtype = object)
vertex = 0
M = valence[vertex, 0]
r_trig_term = np.arange(M)
r_xi_trig_term = np.cos((r_trig_term * 2 * np.pi) / M)
r_eta_trig_term = np.sin((r_trig_term * 2 * np.pi) / M)
if M > 4:
    r_xi_eta_trig_term = r_xi_trig_term * r_eta_trig_term
    # print(r_xi_trig_term)
    # print(r_eta_trig_term)
    # print(r_xi_eta_trig_term)
    # r_xi_eta = (8 / M) * np.dot(r_xi_1, xi_eta_trig_term)


########################################

# # get r_xi_eta for M>=5
# if v > 4:
#
#     print(" ")
#     xi_eta_trig_term = cos_theta_m * sin_theta_m
#     print("cos_theta_m", cos_theta_m)
#     print("sin_theta_m", sin_theta_m)
#     print("xi_eta_trig_term", xi_eta_trig_term)
#     print("r_xi_1: ", r_xi_1)
#     r_xi_eta = (8 / v) * np.dot(r_xi_1, xi_eta_trig_term)
#     print("r_xi_eta: ", r_xi_eta)
#
# # get r_xi_eta for M=4
# else:
#     print(" ")
#     print("neighbours size", P1neighbours.size)
#     print("diagonals size", P1diagonals.size)
#     v_hat = P1neighbours.size + P1diagonals.size
#     print("v_hat size", v_hat)
#     print("neighbours ", P1neighbours)
#     print("diagonals ", P1diagonals)
#     neighbours_and_diagonals = np.sort(np.concatenate((P1neighbours, P1diagonals)))
#     print("neighbours_and_diagonals ", neighbours_and_diagonals)
#     print("neighbours_and_diagonals size ", neighbours_and_diagonals.size)
#     r_xi_eta_1 = np.zeros([3, v_hat])
#     for i, P in enumerate(neighbours_and_diagonals):
#         r_xi_eta_1[:, i] = surfacePoints[P]
#     print("r_xi_eta_1", r_xi_eta_1)
#
#     cos_theta_hat_m = np.zeros([v_hat, 1])
#     sin_theta_hat_m = np.zeros([v_hat, 1])
#     for i in range(v_hat):
#         cos_theta_hat_m[i] = np.cos(((i + 0.5) * 2 * np.pi) / v_hat)
#         sin_theta_hat_m[i] = np.sin(((i + 0.5) * 2 * np.pi) / v_hat)
#     print("cos_theta_hat_m", cos_theta_hat_m)
#     print("sin_theta_hat_m", sin_theta_hat_m)
#     xi_eta_trig_term = cos_theta_hat_m * sin_theta_hat_m
#     print("sum: ", np.sum(xi_eta_trig_term))
#
#     r_xi_eta = (2 / v_hat) * np.dot(r_xi_eta_1, xi_eta_trig_term)
#     print("r_xi_eta: ", r_xi_eta)

#######################################
