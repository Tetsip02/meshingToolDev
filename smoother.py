import meshio
import numpy as np
# import math
from helpers import *
# import json
import pickle

np.set_printoptions(precision=16)

surface_triangles_ini, surface_quads_ini, surface_points = read_surface("./testGeometries/sphere.obj")

##########################################################################################

with open("./dataOutput/V2_triFaceIndices.txt", "rb") as fp:   # Unpickling
    triFaceIndices = pickle.load(fp)
with open("./dataOutput/V2_quadFaceIndices.txt", "rb") as fp:   # Unpickling
    quadFaceIndices = pickle.load(fp)
with open("./dataOutput/V2_directValence.txt", "rb") as fp:   # Unpickling
    directValence = pickle.load(fp)
with open("./dataOutput/V2_directNeighbours.txt", "rb") as fp:   # Unpickling
    directNeighbours = pickle.load(fp)
with open("./dataOutput/V2_simpleDiagonals.txt", "rb") as fp:   # Unpickling
    simpleDiagonals = pickle.load(fp)
with open("./dataOutput/V2_interDiagonals.txt", "rb") as fp:   # Unpickling
    interDiagonals = pickle.load(fp)

######################################################################

#change orientation
surface_triangles = surface_triangles_ini.copy()
surface_triangles[:,[1, 2]] = surface_triangles[:,[2, 1]]
surface_quads = surface_quads_ini.copy()
surface_quads[:,[1, 3]] = surface_quads[:,[3, 1]]

######################################

nPoints = len(surface_points)
nTriFaces = len(surface_triangles)
nQuadFaces = len(surface_quads)
nFaces = nTriFaces + nQuadFaces

######################################################

level2_triangles = surface_triangles + nPoints
level2_quads = surface_quads + nPoints
level3_triangles = level2_triangles + nPoints
level3_quads = level2_quads + nPoints

########################################################################

# # build face normals
# triNormal = np.ndarray((nTriFaces, 3), dtype = np.float64)
# quadNormal = np.ndarray((nQuadFaces, 3), dtype = np.float64)
# for i, triFace in enumerate(surface_triangles):
#     coor = surface_points[triFace, :]
#     triNormal[i, :] = np.cross(coor[1] - coor[0], coor[2] - coor[0])
# for i, quadFace in enumerate(surface_quads):
#     coor = surface_points[quadFace, :]
#     quadNormal[i, :] = np.cross(coor[1] - coor[0], coor[2] - coor[0])
#
# # build vertex normals
# vertexNormal = np.zeros([nPoints, 3], dtype = np.float64)
# for vertex, (triNeighbours, quadNeighbours) in enumerate(zip(triFaceIndices, quadFaceIndices)):
#     vertexNormal[vertex] += triNormal[triNeighbours].sum(axis = 0) + quadNormal[quadNeighbours].sum(axis = 0)

#########################################################

######################################
def get_layer_normals(points, triangles, quads, tri_neighbours, quad_neighbours):
    # build face normals
    triNormal = np.ndarray((nTriFaces, 3), dtype = np.float64)
    quadNormal = np.ndarray((nQuadFaces, 3), dtype = np.float64)
    for i, triFace in enumerate(triangles):
        coor = points[triFace, :]
        triNormal[i, :] = np.cross(coor[1] - coor[0], coor[2] - coor[0])
    for i, quadFace in enumerate(quads):
        coor = surface_points[quadFace, :]
        quadNormal[i, :] = np.cross(coor[1] - coor[0], coor[2] - coor[0])
    # build vertex normals
    vertexNormal = np.zeros([points.shape[0], 3], dtype = np.float64)
    for vertex, (triNeighbours, quadNeighbours) in enumerate(zip(tri_neighbours, quad_neighbours)):
        vertexNormal[vertex] += triNormal[triNeighbours].sum(axis = 0) + quadNormal[quadNeighbours].sum(axis = 0)
    # normalize
    vertexNormal_norm = vertexNormal / np.linalg.norm(vertexNormal, axis = 1)[: , None]

    return vertexNormal, vertexNormal_norm

#############################################################

def get_derivatives(level2, level1_normals, level2_normals, d1, d2, directValence, directNeighbours, simpleDiagonals, interDiagonals):
    zeta = (level1_normals * d1 + level2_normals * d2) / (d1 + d2)
    zeta_zeta = (1 * (level2_normals - level1_normals)) / (d1 + d2) ## double check if the 2 multiplier if correct
    xi = np.ndarray((nPoints, 3), dtype = np.float64)
    eta = np.ndarray((nPoints, 3), dtype = np.float64)
    xi_xi = np.ndarray((nPoints, 3), dtype = np.float64)
    eta_eta = np.ndarray((nPoints, 3), dtype = np.float64)
    xi_eta = np.ndarray((nPoints, 3), dtype = np.float64)
    for vertex, val in enumerate(directValence):
        if val > 4:
            neigh = level2[directNeighbours[vertex]]
            trig_base = np.arange(val)
            xi_trig = np.transpose(np.cos((trig_base * 2 * np.pi) / val)[:, None])
            eta_trig = np.transpose(np.sin((trig_base * 2 * np.pi) / val)[:, None])
            xi_xi_trig = 4 * np.square(xi_trig) - 1
            eta_eta_trig = 4 * np.square(eta_trig) - 1
            xi_eta_trig = xi_trig * eta_trig
            xi[vertex, :] = (2 / val) * np.dot(xi_trig, neigh)
            eta[vertex, :] = (2 / val) * np.dot(eta_trig, neigh)
            xi_xi[vertex, :] = (2 / val) * np.dot(xi_xi_trig, neigh)
            eta_eta[vertex, :] = (2 / val) * np.dot(eta_eta_trig, neigh)
            xi_eta[vertex, :] = (8 / val) * np.dot(xi_eta_trig, neigh)
        elif val == 3:
            neigh = level2[directNeighbours[vertex]]
            if interDiagonals[vertex]:
                interNeigh = np.mean(np.array(level2[interDiagonals[vertex], :]), axis = 1)
                neigh = np.concatenate((neigh, interNeigh))
            M = neigh.shape[0]
            trig_base = np.arange(M)
            xi_trig = np.transpose(np.cos((trig_base * 2 * np.pi) / M)[:, None])
            eta_trig = np.transpose(np.sin((trig_base * 2 * np.pi) / M)[:, None])
            xi_xi_trig = 4 * np.square(xi_trig) - 1
            eta_eta_trig = 4 * np.square(eta_trig) - 1
            xi_eta_trig = xi_trig * eta_trig
            xi[vertex, :] = (2 / M) * np.dot(xi_trig, neigh)
            eta[vertex, :] = (2 / M) * np.dot(eta_trig, neigh)
            xi_xi[vertex, :] = (2 / M) * np.dot(xi_xi_trig, neigh)
            eta_eta[vertex, :] = (2 / M) * np.dot(eta_eta_trig, neigh)
            xi_eta[vertex, :] = (8 / M) * np.dot(xi_eta_trig, neigh)
        elif val == 4:
            neigh = level2[directNeighbours[vertex]]
            neigh_hat = neigh.copy()
            if simpleDiagonals[vertex]:
                simpleNeigh = level2[simpleDiagonals[vertex]]
                neigh_hat = np.concatenate((neigh_hat, simpleNeigh))
            if interDiagonals[vertex]:
                interNeigh = np.mean(np.array(level2[interDiagonals[vertex], :]), axis = 1)
                neigh_hat = np.concatenate((neigh_hat, interNeigh))
            eta[vertex, :] = (2 / val) * (neigh[0] - neigh[2])
            eta[vertex, :] = (2 / val) * (neigh[1] - neigh[3])
            xi_xi[vertex, :] = neigh[0] + neigh[2]
            xi_xi[vertex, :] = neigh[1] + neigh[3]
            M = 8
            trig_base = np.arange(M)
            xi_eta_trig = np.cos(((trig_base + 0.5) * 2 * np.pi) / M) * np.sin(((trig_base + 0.5) * 2 * np.pi) / M)
            xi_eta[vertex, :] = (2 / M) * np.dot(xi_eta_trig, neigh_hat)

    return xi, eta, xi_xi, eta_eta, xi_eta, zeta, zeta_zeta

#######################################################################

def get_tensors(xi, eta, xi_xi, eta_eta, xi_eta, zeta, zeta_zeta):
    g_11 = xi[: , 0] ** 2 + xi[: , 1] ** 2 + xi[: , 2] ** 2
    g_22 = eta[: , 0] ** 2 + eta[: , 1] ** 2 + eta[: , 2] ** 2
    g_33 = zeta[: , 0] ** 2 + zeta[: , 1] ** 2 + zeta[: , 2] ** 2
    g_12 = xi[: , 0] * eta[: , 0] + xi[: , 1] * eta[: , 1] + xi[: , 2] * eta[: , 2]
    g = xi[: , 0] * (eta[: , 1] * zeta[: , 2] - eta[: , 2] * zeta[: , 1]) - xi[: , 1] * (eta[: , 0] * zeta[: , 2] - eta[: , 2] * zeta[: , 0]) + xi[: , 2] * (eta[: , 0] * zeta[: , 1] - eta[: , 1] * zeta[: , 0])
    g_square = g ** 2
    c1 = (g_22 * g_33) / g_square
    c2 = (g_11 * g_33) / g_square
    c3 = -2 * ((g_12 * g_33) / g_square)
    c4 = (g_11 * g_22 - g_12 ** 2) / g_square

    return g_11, g_22, g_33, g_12, g_square, c1, c2, c3, c4


########################################################################

level1_normals, level1_normals_norm = get_layer_normals(surface_points, surface_triangles, surface_quads, triFaceIndices, quadFaceIndices)

#########################################################


# temporary reference mesh
# level1 = surface_points.copy()
d1 = 0.05
d2 = 0.07
level2 = surface_points + d1 * level1_normals_norm
level2_normals, level2_normals_norm = get_layer_normals(level2, surface_triangles, surface_quads, triFaceIndices, quadFaceIndices)
level3 = level2 + d2 * level2_normals_norm

##############################################################

refMesh_points = np.concatenate((surface_points, level2, level3))
voxels1 = [["wedge", np.concatenate((surface_triangles, level2_triangles),axis=1)], ["hexahedron", np.concatenate((surface_quads, level2_quads),axis=1)]]
voxels2 = [["wedge", np.concatenate((level2_triangles, level3_triangles),axis=1)], ["hexahedron", np.concatenate((level2_quads, level3_quads),axis=1)]]
voxels1and2 = voxels1
voxels1and2.extend(voxels2)
V6_referenceMesh = meshio.Mesh(points = refMesh_points, cells = voxels1and2)
meshio.write("./output/V6_referenceMesh.vtk", V6_referenceMesh, file_format="vtk", binary=False)
# meshio.write("./output/ref_mesh_test2.msh", V5_mesh, file_format="gmsh22", binary=False)

###############################################################

xi, eta, xi_xi, eta_eta, xi_eta, zeta, zeta_zeta = get_derivatives(level2, level1_normals, level2_normals, d1, d2, directValence, directNeighbours, simpleDiagonals, interDiagonals)

#############################################################

g_11, g_22, g_33, g_12, g_square, c1, c2, c3, c4 = get_tensors(xi, eta, xi_xi, eta_eta, xi_eta, zeta, zeta_zeta)

print(g_square)

############################################################

# n_it = 5
# for it in range(n_it)
#
# level2_it1 = (c1[: , None] * xi_xi + c2[: , None] * eta_eta + c3[: , None] * xi_eta + c4[: , None] * zeta_zeta) / (2 * (c1[: , None] + c2[: , None]))
# points_it1 = np.concatenate((level1, level2_it1))
# points_it1 = points_it1.astype('float64')
# points_it0 = np.concatenate((level1, level2))

# print(g_11[0])
# print(g_22[0])
# print(g_33[0])
# print(g_12[0])
# print(g[0])
# print(g_square[0])
# print(c1[0])
# print(c2[0])
# print(c3[0])
# print(c4[0])
# print(xi_xi[0])
# print(eta_eta[0])
# print(xi_eta[0])
# print(zeta_zeta[0])

##############################################################

layer1_points = np.concatenate((surface_points, level2_it1))
# layer1_points = layer1_points.astype('float64')
voxels1_it1 = [["wedge", np.concatenate((surface_triangles, level2_triangles),axis=1)], ["hexahedron", np.concatenate((surface_quads, level2_quads),axis=1)]]
voxels2 = [["wedge", np.concatenate((level2_triangles, level3_triangles),axis=1)], ["hexahedron", np.concatenate((level2_quads, level3_quads),axis=1)]]
# voxels1and2 = voxels1
# voxels1and2.extend(voxels2)
level2SurfaceV = [["triangle", surface_triangles], ["quad", surface_quads]]
level2Surface = meshio.Mesh(points = level2_it1, cells = level2SurfaceV)
V5_mesh_layer1 = meshio.Mesh(points = layer1_points, cells = voxels1_it1)
meshio.write("./output/V5_mesh_layer1.vtk", V5_mesh_layer1, file_format="vtk", binary=False)
meshio.write("./output/V5_mesh_layer1.msh", V5_mesh_layer1, file_format="gmsh22", binary=False)
meshio.write("./output/V5_mesh_level2Surface.vtk", level2Surface, file_format="vtk", binary=False)

# #############################################################
# #iteration 2
# # level2 = surface_points + d1 * vertexNormal_norm
# level2_triNormal2 = np.ndarray((nTriFaces, 3), dtype = np.float64)
# level2_quadNormal2 = np.ndarray((nQuadFaces, 3), dtype = np.float64)
# level2_vertexNormal2 = np.zeros([nPoints, 3], dtype = np.float64)
# for i, triFace in enumerate(surface_triangles):
#     coor = level2_it1[triFace, :]
#     level2_triNormal2[i, :] = np.cross(coor[1] - coor[0], coor[2] - coor[0])
# for i, quadFace in enumerate(surface_quads):
#     coor = level2_it1[quadFace, :]
#     level2_quadNormal2[i, :] = np.cross(coor[1] - coor[0], coor[2] - coor[0])
# for vertex, (triNeighbours, quadNeighbours) in enumerate(zip(triFaceIndices, quadFaceIndices)):
#     level2_vertexNormal2[vertex] += level2_triNormal[triNeighbours].sum(axis = 0) + level2_quadNormal[quadNeighbours].sum(axis = 0)
# level2_vertexNormal_norm2 = level2_vertexNormal / np.linalg.norm(vertexNormal, axis = 1)[: , None]
# level3_it1 = level2_it1 + d2 * level2_vertexNormal_norm2
#
# zeta = (vertexNormal * d1 + level2_vertexNormal2 * d2) / (d1 + d2)
# zeta_zeta = (1 * (level2_vertexNormal2 - vertexNormal)) / (d1 + d2)
#
# xi = np.ndarray((nPoints, 3), dtype = np.float64)
# eta = np.ndarray((nPoints, 3), dtype = np.float64)
# xi_xi = np.ndarray((nPoints, 3), dtype = np.float64)
# eta_eta = np.ndarray((nPoints, 3), dtype = np.float64)
# xi_eta = np.ndarray((nPoints, 3), dtype = np.float64)
#
#
# for vertex, val in enumerate(directValence):
#     if val > 4:
#         neigh = level2_it1[directNeighbours[vertex]]
#         trig_base = np.arange(val)
#         xi_trig = np.transpose(np.cos((trig_base * 2 * np.pi) / val)[:, None])
#         eta_trig = np.transpose(np.sin((trig_base * 2 * np.pi) / val)[:, None])
#         xi_xi_trig = 4 * np.square(xi_trig) - 1
#         eta_eta_trig = 4 * np.square(eta_trig) - 1
#         xi_eta_trig = xi_trig * eta_trig
#         xi[vertex, :] = (2 / val) * np.dot(xi_trig, neigh)
#         eta[vertex, :] = (2 / val) * np.dot(eta_trig, neigh)
#         xi_xi[vertex, :] = (2 / val) * np.dot(xi_xi_trig, neigh)
#         eta_eta[vertex, :] = (2 / val) * np.dot(eta_eta_trig, neigh)
#         xi_eta[vertex, :] = (8 / val) * np.dot(xi_eta_trig, neigh)
#     elif val == 3:
#         neigh = level2_it1[directNeighbours[vertex]]
#         if interDiagonals[vertex]:
#             interNeigh = np.mean(np.array(level2_it1[interDiagonals[vertex], :]), axis = 1)
#             neigh = np.concatenate((neigh, interNeigh))
#         M = neigh.shape[0]
#         trig_base = np.arange(M)
#         xi_trig = np.transpose(np.cos((trig_base * 2 * np.pi) / M)[:, None])
#         eta_trig = np.transpose(np.sin((trig_base * 2 * np.pi) / M)[:, None])
#         xi_xi_trig = 4 * np.square(xi_trig) - 1
#         eta_eta_trig = 4 * np.square(eta_trig) - 1
#         xi_eta_trig = xi_trig * eta_trig
#         xi[vertex, :] = (2 / M) * np.dot(xi_trig, neigh)
#         eta[vertex, :] = (2 / M) * np.dot(eta_trig, neigh)
#         xi_xi[vertex, :] = (2 / M) * np.dot(xi_xi_trig, neigh)
#         eta_eta[vertex, :] = (2 / M) * np.dot(eta_eta_trig, neigh)
#         xi_eta[vertex, :] = (8 / M) * np.dot(xi_eta_trig, neigh)
#     elif val == 4:
#         neigh = level2_it1[directNeighbours[vertex]]
#         neigh_hat = neigh.copy()
#         if simpleDiagonals[vertex]:
#             simpleNeigh = level2_it1[simpleDiagonals[vertex]]
#             neigh_hat = np.concatenate((neigh_hat, simpleNeigh))
#         if interDiagonals[vertex]:
#             interNeigh = np.mean(np.array(level2_it1[interDiagonals[vertex], :]), axis = 1)
#             neigh_hat = np.concatenate((neigh_hat, interNeigh))
#         eta[vertex, :] = (2 / val) * (neigh[0] - neigh[2])
#         eta[vertex, :] = (2 / val) * (neigh[1] - neigh[3])
#         xi_xi[vertex, :] = neigh[0] + neigh[2]
#         xi_xi[vertex, :] = neigh[1] + neigh[3]
#         M = 8
#         trig_base = np.arange(M)
#         xi_eta_trig = np.cos(((trig_base + 0.5) * 2 * np.pi) / M) * np.sin(((trig_base + 0.5) * 2 * np.pi) / M)
#         xi_eta[vertex, :] = (2 / M) * np.dot(xi_eta_trig, neigh_hat)
#
# g_11 = xi[: , 0] ** 2 + xi[: , 1] ** 2 + xi[: , 2] ** 2
# g_22 = eta[: , 0] ** 2 + eta[: , 1] ** 2 + eta[: , 2] ** 2
# g_33 = zeta[: , 0] ** 2 + zeta[: , 1] ** 2 + zeta[: , 2] ** 2
# g_12 = xi[: , 0] * eta[: , 0] + xi[: , 1] * eta[: , 1] + xi[: , 2] * eta[: , 2]
# g = xi[: , 0] * (eta[: , 1] * zeta[: , 2] - eta[: , 2] * zeta[: , 1]) - xi[: , 1] * (eta[: , 0] * zeta[: , 2] - eta[: , 2] * zeta[: , 0]) + xi[: , 2] * (eta[: , 0] * zeta[: , 1] - eta[: , 1] * zeta[: , 0])
# g_square = g ** 2
# c1 = (g_22 * g_33) / g_square
# c2 = (g_11 * g_33) / g_square
# c3 = -2 * ((g_12 * g_33) / g_square)
# c4 = (g_11 * g_22 - g_12 ** 2) / g_square
# level2_it2 = (c1[: , None] * xi_xi + c2[: , None] * eta_eta + c3[: , None] * xi_eta + c4[: , None] * zeta_zeta) / (2 * (c1[: , None] + c2[: , None]))
#
# layer1_points2 = np.concatenate((surface_points, level2_it2))
# # layer1_points = layer1_points.astype('float64')
# voxels1_it2 = [["wedge", np.concatenate((surface_triangles, level2_triangles),axis=1)], ["hexahedron", np.concatenate((surface_quads, level2_quads),axis=1)]]
# # voxels2 = [["wedge", np.concatenate((level2_triangles, level3_triangles),axis=1)], ["hexahedron", np.concatenate((level2_quads, level3_quads),axis=1)]]
# # voxels1and2 = voxels1
# # voxels1and2.extend(voxels2)
# # level2SurfaceV = [["triangle", surface_triangles], ["quad", surface_quads]]
# # level2Surface = meshio.Mesh(points = level2_it1, cells = level2SurfaceV)
# V5_mesh_layer1_it2 = meshio.Mesh(points = layer1_points2, cells = voxels1_it2)
# meshio.write("./output/V5_mesh_layer1_it2.vtk", V5_mesh_layer1_it2, file_format="vtk", binary=False)
# # meshio.write("./output/V5_mesh_layer1.msh", V5_mesh_layer1, file_format="gmsh22", binary=False)
# # meshio.write("./output/V5_mesh_level2Surface.vtk", level2Surface, file_format="vtk", binary=False)
# ###############################################################
# control functions
# CFxi = np.ndarray((nPoints, 3), dtype = np.float64)
# CFeta = np.ndarray((nPoints, 3), dtype = np.float64)
# CFxi_xi = np.ndarray((nPoints, 3), dtype = np.float64)
# CFeta_eta = np.ndarray((nPoints, 3), dtype = np.float64)
# CFxi_eta = np.ndarray((nPoints, 3), dtype = np.float64)
#
# for vertex, val in enumerate(directValence):
#     if val > 4:
#         neigh = surface_points[directNeighbours[vertex]]
#         trig_base = np.arange(val)
#         xi_trig = np.transpose(np.cos((trig_base * 2 * np.pi) / val)[:, None])
#         eta_trig = np.transpose(np.sin((trig_base * 2 * np.pi) / val)[:, None])
#         xi_xi_trig = 4 * np.square(xi_trig) - 1
#         eta_eta_trig = 4 * np.square(eta_trig) - 1
#         xi_eta_trig = xi_trig * eta_trig
#         CFxi[vertex, :] = (2 / val) * np.dot(xi_trig, neigh)
#         CFeta[vertex, :] = (2 / val) * np.dot(eta_trig, neigh)
#         CFxi_xi[vertex, :] = (2 / val) * np.dot(xi_xi_trig, neigh)
#         CFeta_eta[vertex, :] = (2 / val) * np.dot(eta_eta_trig, neigh)
#         CFxi_eta[vertex, :] = (8 / val) * np.dot(xi_eta_trig, neigh)
#     elif val == 3:
#         neigh = surface_points[directNeighbours[vertex]]
#         if interDiagonals[vertex]:
#             interNeigh = np.mean(np.array(surface_points[interDiagonals[vertex], :]), axis = 1)
#             neigh = np.concatenate((neigh, interNeigh))
#         M = neigh.shape[0]
#         trig_base = np.arange(M)
#         xi_trig = np.transpose(np.cos((trig_base * 2 * np.pi) / M)[:, None])
#         eta_trig = np.transpose(np.sin((trig_base * 2 * np.pi) / M)[:, None])
#         xi_xi_trig = 4 * np.square(xi_trig) - 1
#         eta_eta_trig = 4 * np.square(eta_trig) - 1
#         xi_eta_trig = xi_trig * eta_trig
#         CFxi[vertex, :] = (2 / M) * np.dot(xi_trig, neigh)
#         CFeta[vertex, :] = (2 / M) * np.dot(eta_trig, neigh)
#         CFxi_xi[vertex, :] = (2 / M) * np.dot(xi_xi_trig, neigh)
#         CFeta_eta[vertex, :] = (2 / M) * np.dot(eta_eta_trig, neigh)
#         CFxi_eta[vertex, :] = (8 / M) * np.dot(xi_eta_trig, neigh)
#     elif val == 4:
#         neigh = surface_points[directNeighbours[vertex]]
#         neigh_hat = neigh.copy()
#         if simpleDiagonals[vertex]:
#             simpleNeigh = surface_points[simpleDiagonals[vertex]]
#             neigh_hat = np.concatenate((neigh_hat, simpleNeigh))
#         if interDiagonals[vertex]:
#             interNeigh = np.mean(np.array(surface_points[interDiagonals[vertex], :]), axis = 1)
#             neigh_hat = np.concatenate((neigh_hat, interNeigh))
#         CFeta[vertex, :] = (2 / val) * (neigh[0] - neigh[2])
#         CFeta[vertex, :] = (2 / val) * (neigh[1] - neigh[3])
#         CFxi_xi[vertex, :] = neigh[0] + neigh[2]
#         CFxi_xi[vertex, :] = neigh[1] + neigh[3]
#         M = 8
#         trig_base = np.arange(M)
#         xi_eta_trig = np.cos(((trig_base + 0.5) * 2 * np.pi) / M) * np.sin(((trig_base + 0.5) * 2 * np.pi) / M)
#         CFxi_eta[vertex, :] = (2 / M) * np.dot(xi_eta_trig, neigh_hat)

# print(xi[0, :])
# print(eta[0, :])
# print(xi_xi[0, :])
# print(eta_eta[0, :])
# print(xi_eta[0, :])
# theta = (-0.5) * ((xi_xi * xi) / g_33[: , None])
# print(xi.shape)
# print(xi_xi.shape)
# print(g_33.shape)
# print(theta)
# print(theta.shape)
# ##########################################################

# ####################################################################




####################################################
