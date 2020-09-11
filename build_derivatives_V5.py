import meshio
import numpy as np
# import math
from helpers import *
# import json
import pickle

np.set_printoptions(precision=16)

surface_triangles, surface_quads, surface_points = read_surface("./testGeometries/sphere.obj")
# d1 = 0.1
# d2 = 0.12
# level1, level2, level3 = build_ref_mesh_points(surface_points, surface_triangles, surface_quads, d1, d2, 1)
# nSurfacePoints = len(surface_points)
# level1_triangles = surface_triangles
# level1_quads = surface_quads
# level2_triangles = level1_triangles + nSurfacePoints
# level2_quads = level1_quads + nSurfacePoints
# level3_triangles = level2_triangles + nSurfacePoints
# level3_quads = level2_quads + nSurfacePoints
# allPoints = np.concatenate((level1, level2, level3))

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

nPoints = len(surface_points)
nTriFaces = len(surface_triangles)
nQuadFaces = len(surface_quads)
nFaces = nTriFaces + nQuadFaces

######################################

# build face normals
triNormal = np.ndarray((nTriFaces, 3), dtype = np.float64)
quadNormal = np.ndarray((nQuadFaces, 3), dtype = np.float64)
for i, triFace in enumerate(surface_triangles):
    coor = surface_points[triFace, :]
    triNormal[i, :] = np.cross(coor[1] - coor[0], coor[2] - coor[0])
for i, quadFace in enumerate(surface_quads):
    coor = surface_points[quadFace, :]
    quadNormal[i, :] = np.cross(coor[1] - coor[0], coor[2] - coor[0])

# build vertex normals
vertexNormal = np.zeros([nPoints, 3], dtype = np.float64)
for vertex, (triNeighbours, quadNeighbours) in enumerate(zip(triFaceIndices, quadFaceIndices)):
    vertexNormal[vertex] += triNormal[triNeighbours].sum(axis = 0) + quadNormal[quadNeighbours].sum(axis = 0)

#########################################################

# create normalized normals
triNormal_norm = triNormal / np.linalg.norm(triNormal, axis = 1)[: , None]

quadNormal_norm = quadNormal / np.linalg.norm(quadNormal, axis = 1)[: , None]

vertexNormal_norm = vertexNormal / np.linalg.norm(vertexNormal, axis = 1)[: , None]

###################################################

# reference mesh extrusion by vertex normals
d = 0.05
nextLevel = surface_points + d * vertexNormal_norm
# print(nextLevel)

################################################

# temporary reference mesh
# level1 = surface_points.copy()
d1 = 0.05
d2 = 0.05
level2 = surface_points + d1 * vertexNormal_norm
level2_triNormal = np.ndarray((nTriFaces, 3), dtype = np.float64)
level2_quadNormal = np.ndarray((nQuadFaces, 3), dtype = np.float64)
level2_vertexNormal = np.zeros([nPoints, 3], dtype = np.float64)
for i, triFace in enumerate(surface_triangles):
    coor = level2[triFace, :]
    level2_triNormal[i, :] = np.cross(coor[1] - coor[0], coor[2] - coor[0])
for i, quadFace in enumerate(surface_quads):
    coor = level2[quadFace, :]
    level2_quadNormal[i, :] = np.cross(coor[1] - coor[0], coor[2] - coor[0])
for vertex, (triNeighbours, quadNeighbours) in enumerate(zip(triFaceIndices, quadFaceIndices)):
    level2_vertexNormal[vertex] += level2_triNormal[triNeighbours].sum(axis = 0) + level2_quadNormal[quadNeighbours].sum(axis = 0)
level3 = level2 + d2 * vertexNormal_norm

##############################################################

level2_triangles = surface_triangles + nPoints
level2_quads = surface_quads + nPoints
level3_triangles = level2_triangles + nPoints
level3_quads = level2_quads + nPoints
refMesh_points = np.concatenate((surface_points, level2, level3))
voxels1 = [["wedge", np.concatenate((surface_triangles, level2_triangles),axis=1)], ["hexahedron", np.concatenate((surface_quads, level2_quads),axis=1)]]
voxels2 = [["wedge", np.concatenate((level2_triangles, level3_triangles),axis=1)], ["hexahedron", np.concatenate((level2_quads, level3_quads),axis=1)]]
voxels1and2 = voxels1
voxels1and2.extend(voxels2)
V5_mesh = meshio.Mesh(points = refMesh_points, cells = voxels1and2)
# meshio.write("./output/V5_mesh.vtk", V5_mesh, file_format="vtk", binary=False)
# meshio.write("./output/ref_mesh_test2.msh", V5_mesh, file_format="gmsh22", binary=False)

###############################################################

r_xi = np.ndarray((nPoints, 3), dtype = np.float64)
r_eta = np.ndarray((nPoints, 3), dtype = np.float64)
r_xi_xi = np.ndarray((nPoints, 3), dtype = np.float64)
r_eta_eta = np.ndarray((nPoints, 3), dtype = np.float64)
r_xi_eta = np.ndarray((nPoints, 3), dtype = np.float64)

vertex = 0
# cos_term = np.cos(np.arange(max(directValence)) * 2 * np.pi)
# sin_term = np.sin(np.arange(max(directValence)) * 2 * np.pi)
print(directNeighbours[vertex])
# print(directValence[vertex])
# print(surface_points.dtype)
# print(level2.dtype)
# # print(level2[[1, 2]])
# print(level2[directNeighbours[vertex]])

M = directValence[vertex]
neigh = level2[directNeighbours[vertex]]
# neigh = np.transpose(level2[directNeighbours[vertex]])
val = directValence[vertex]
trig = np.transpose(np.cos((np.arange(val) * 2 * np.pi) / val)[:, None])
# test = (2 / directValence[vertex]) * np.dot(level2[directNeighbours[vertex]], np.cos((np.arange(directValence[vertex]) * 2 * np.pi) / directValence[vertex]))
print(neigh.shape)
print(trig.shape)
test = (2 / M) * np.dot(trig, neigh)
print(test.shape)

# for vertex in range(nPoints):
#     M = valence[vertex, 0]
#     r_trig_term = np.arange(M)
#     r_xi_trig_term = np.cos((r_trig_term * 2 * np.pi) / M)
#     r_eta_trig_term = np.sin((r_trig_term * 2 * np.pi) / M)
#
#     neighbourlist = directNeighbours[vertex, :]
#
#     neighbourCoor = np.ndarray((3, M), dtype = object)
#     for m in range(M):
#         neighbourCoor[: , m] = level2[neighbourlist[m], :]
#
#     r_xi[vertex, :] = (2 / M) * np.dot(neighbourCoor, r_xi_trig_term)
#     r_eta[vertex, :] = (2 / M) * np.dot(neighbourCoor, r_eta_trig_term)
#
#     if M > 4:
#         # evaluate r_xi_xi
#         r_xi_xi_trig_term = 4 * np.square(r_xi_trig_term) - 1
#         r_xi_xi[vertex, :] = (2 / M) * np.dot(neighbourCoor, r_xi_xi_trig_term)
#
#         # evaluate r_eta_eta
#         r_eta_eta_trig_term = 4 * np.square(r_eta_trig_term) - 1
#         r_eta_eta[vertex, :] = (2 / M) * np.dot(neighbourCoor, r_eta_eta_trig_term)
#
#         # evaluate r_xi_eta
#         r_xi_eta_trig_term = r_xi_trig_term * r_eta_trig_term
#         r_xi_eta[vertex, :] = (8 / M) * np.dot(neighbourCoor, r_eta_trig_term)
#
#     elif M == 4:
#         # evaluate r_xi_xi
#         r_xi_xi[vertex, :] = neighbourCoor[:, 0] + neighbourCoor[:, 2]
#
#         # evaluate r_eta_eta
#         r_eta_eta[vertex, :] = neighbourCoor[:, 1] + neighbourCoor[:, 3]
#
#         # evaluate r_xi_eta
#         M_hat = M + nDiagonals[vertex, 0]
#         neighbours_hat = np.concatenate(([directNeighbours[vertex, [0, 1, 2, 3]], diagonals[vertex, 0:nDiagonals[vertex, 0]]]))
#         neighbour_hatCoor = np.ndarray((3, M_hat), dtype = object)
#         for m_hat in range(M_hat):
#             neighbour_hatCoor[: , m_hat] = level2[neighbours_hat[m_hat], :]
#         r_trig_term = np.arange(M_hat)
#         r_xi_eta_trig_term = np.cos(((r_trig_term + 0.5) * 2 * np.pi) / M_hat) * np.sin(((r_trig_term + 0.5) * 2 * np.pi) / M_hat)
#         r_xi_eta[vertex, :] = (2 / M_hat) * np.dot(neighbour_hatCoor, r_xi_eta_trig_term)


# ##########################################################
#
# # extrude points / build level 2
# print(surface_points.shape)
# # nextLevel = np.ndarray(surface_points.shape, dtype = np.float64)
# nextLevel = surface_points.copy()
# d = 0.05
# print(triNormal_norm.shape)
#
# # for i, triFace in enumerate(surface_triangles):
# #     # coor = surface_points[triFace]
# #     # n = triNormal_norm[triFace]
# #     # next = surface_points[triFace] + d * triNormal_norm[triFace]
# #     # print(triFace)
# #     nextLevel[triFace] = surface_points[triFace] + d * triNormal_norm[i]
# #
# # print(nextLevel[0:100])
#
# triFaceTest = np.array([3, 4, 5])
# print(triFaceTest)
# print("point 3", surface_points[3])
# print("point 4", surface_points[4])
# print("point 5", surface_points[5])
# print("face normal 1", triNormal_norm[1])
# coor = surface_points[triFaceTest]
# print(coor)
# n = triNormal_norm[1] ###this can't be right
# print(n)
# next = coor + d * n
# print("next", next)
# print("next level coor", nextLevel[triFaceTest])
# print("surface coor", surface_points[triFaceTest])
# print(nextLevel[triFaceTest[0]])
# for i in [0, 1, 2]:
#     print(i)
#     print(nextLevel[triFaceTest[i]] == surface_points[triFaceTest[i]])
#     # if nextLevel[triFaceTest[i]] == surface_points[triFaceTest[i]]:
# # if nextLevel[triFaceTest] == surface_points[triFaceTest]:
# #     nextLevel[triFaceTest] = next
# # print(nextLevel[triFaceTest] == surface_points[triFaceTest])
# # print(np.all(nextLevel[triFaceTest] - surface_points[triFaceTest], axis = 1))
# # nextLevel[triFaceTest] = next
# # print(nextLevel[0:10])
#
# ####################################################################




####################################################
