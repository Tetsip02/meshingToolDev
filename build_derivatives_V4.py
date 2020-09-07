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
r_zeta = (vertexTable[:, [4, 5, 6]] * d1 + level2VertexTable[:, [4, 5, 6]] * d2) / (d1 + d2)
# print("r_zeta", r_zeta[0:6, :])

######################################

#build r_zeta_zeta
r_zeta_zeta = (level2VertexTable[:, [4, 5, 6]] - vertexTable[:, [4, 5, 6]]) / (d1 + d2)
# print("r_zeta_zeta", r_zeta_zeta[0:6, :])

#######################################

r_xi = np.ndarray((nSurfacePoints, 3), dtype = object)
r_eta = np.ndarray((nSurfacePoints, 3), dtype = object)
r_xi_xi = np.ndarray((nSurfacePoints, 3), dtype = object)
r_eta_eta = np.ndarray((nSurfacePoints, 3), dtype = object)
r_xi_eta = np.ndarray((nSurfacePoints, 3), dtype = object)
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
        r_xi_xi[vertex, :] = (2 / M) * np.dot(neighbourCoor, r_xi_xi_trig_term)
        # r_xi_xi_1 = np.reshape(np.dot(neighbourCoor, r_xi_xi_trig_term), (3, 1))
        # r_xi_xi[vertex, [[0], [1], [2]]] = (2 / M) * Matrix([r_xi_xi_1 - np.sum(r_xi_xi_trig_term) * Matrix([x, y, z])])

        # evaluate r_eta_eta
        r_eta_eta_trig_term = 4 * np.square(r_eta_trig_term) - 1
        r_eta_eta[vertex, :] = (2 / M) * np.dot(neighbourCoor, r_eta_eta_trig_term)
        # r_eta_eta_1 = np.reshape(np.dot(neighbourCoor, r_eta_eta_trig_term), (3, 1))
        # r_eta_eta[vertex, [[0], [1], [2]]] = (2 / M) * Matrix([r_eta_eta_1 - np.sum(r_eta_eta_trig_term) * Matrix([x, y, z])])

        # evaluate r_xi_eta
        r_xi_eta_trig_term = r_xi_trig_term * r_eta_trig_term
        r_xi_eta[vertex, :] = (8 / M) * np.dot(neighbourCoor, r_eta_trig_term)

    elif M == 4:
        # evaluate r_xi_xi
        r_xi_xi[vertex, :] = neighbourCoor[:, 0] + neighbourCoor[:, 2]
        # r_xi_xi_1 = np.reshape((neighbourCoor[:, 0] + neighbourCoor[:, 2]), (3, 1))
        # r_xi_xi[vertex, [[0], [1], [2]]] = Matrix([r_xi_xi_1 - 2 * Matrix([x, y, z])])

        # evaluate r_eta_eta
        r_eta_eta[vertex, :] = neighbourCoor[:, 1] + neighbourCoor[:, 3]
        # r_eta_eta_1 = np.reshape((neighbourCoor[:, 1] + neighbourCoor[:, 3]), (3, 1))
        # r_eta_eta[vertex, [[0], [1], [2]]] = Matrix([r_eta_eta_1 - 2 * Matrix([x, y, z])])

        # evaluate r_xi_eta
        M_hat = M + nDiagonals[vertex, 0]
        neighbours_hat = np.concatenate(([directNeighbours[vertex, [0, 1, 2, 3]], diagonals[vertex, 0:nDiagonals[vertex, 0]]]))
        neighbour_hatCoor = np.ndarray((3, M_hat), dtype = object)
        for m_hat in range(M_hat):
            neighbour_hatCoor[: , m_hat] = level2[neighbours_hat[m_hat], :]
        r_trig_term = np.arange(M_hat)
        r_xi_eta_trig_term = np.cos(((r_trig_term + 0.5) * 2 * np.pi) / M_hat) * np.sin(((r_trig_term + 0.5) * 2 * np.pi) / M_hat)
        r_xi_eta[vertex, :] = (2 / M_hat) * np.dot(neighbour_hatCoor, r_xi_eta_trig_term)

# print(r_xi)
# print(r_eta)
# print(r_xi_xi)
# print(r_eta_eta)
# print(r_xi_eta)
# print(r_zeta)
# print(r_zeta_zeta)


###################################################
# build all tensors

g_11 = r_xi[: , 0] ** 2 + r_xi[: , 1] ** 2 + r_xi[: , 2] ** 2
g_22 = r_eta[: , 0] ** 2 + r_eta[: , 1] ** 2 + r_eta[: , 2] ** 2
g_33 = r_zeta[: , 0] ** 2 + r_zeta[: , 1] ** 2 + r_zeta[: , 2] ** 2
g_12 = r_xi[: , 0] * r_eta[: , 0] + r_xi[: , 1] * r_eta[: , 1] + r_xi[: , 2] * r_eta[: , 2]
g = r_xi[: , 0] * (r_eta[: , 1] * r_zeta[: , 2] - r_eta[: , 2] * r_zeta[: , 1]) - r_xi[: , 1] * (r_eta[: , 0] * r_zeta[: , 2] - r_eta[: , 2] * r_zeta[: , 0]) + r_xi[: , 2] * (r_eta[: , 0] * r_zeta[: , 1] - r_eta[: , 1] * r_zeta[: , 0])
g_square = g ** 2
c1 = (g_22 * g_33) / g_square
c2 = (g_11 * g_33) / g_square
c3 = -2 * ((g_12 * g_33) / g_square)
c4 = (g_11 * g_22 - g_12 ** 2) / g_square
level2_it1 = (c1[: , None] * r_xi_xi + c2[: , None] * r_eta_eta + c3[: , None] * r_xi_eta + c4[: , None] * r_zeta_zeta) / (2 * (c1[: , None] + c2[: , None]))
points_it1 = np.concatenate((level1, level2_it1))
points_it1 = points_it1.astype('float64')
points_it0 = np.concatenate((level1, level2))
print(points_it1.dtype)
print(points_it0.dtype)
print(points_it1[4870:4880, :])
print(points_it0[4870:4880, :])


voxels1 = [["wedge", np.concatenate((level1_triangles, level2_triangles),axis=1)], ["hexahedron", np.concatenate((level1_quads, level2_quads),axis=1)]]
# voxels2 = [["wedge", np.concatenate((level2_triangles, level3_triangles),axis=1)], ["hexahedron", np.concatenate((level2_quads, level3_quads),axis=1)]]
# voxels1and2 = voxels1
# voxels1and2.extend(voxels2)
mesh_test1 = meshio.Mesh(points = points_it1, cells = voxels1)
# mesh_test0 = meshio.Mesh(points = points_it0, cells = voxels1)
# meshio.write("./output/ref_mesh_test2.vtk", mesh_test, file_format="vtk", binary=False)
meshio.write("./output/layer1_it1Smoothing.vtk", mesh_test1, file_format="vtk", binary=False)
# meshio.write("./output/layer1_noSmoothing.vtk", mesh_test0, file_format="vtk", binary=False)

print(vertexTable[305:330, 7])
